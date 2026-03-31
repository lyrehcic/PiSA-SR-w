"""
pisasr_WaveLoRA.py

基于 pisasr.py（原版），结构改造：
  conv 层：双 WaveAdapter（pix / sem）替代 LoRA
  attention 层：保留 LoRA（语义控制，数学上合理）

训练分工：
  Stage1（pix_steps 步）：pix WaveAdapter + attention pix LoRA 训练
  Stage2（之后）：sem WaveAdapter + attention sem LoRA 训练

推理：两次前向，差值机制和原版 PiSASR 完全一致：
  model_pred_full = unet(x)   # pix+sem 都激活
  sem_scale → 0
  model_pred_pix  = unet(x)   # 只有 pix
  sem_scale 恢复
  delta = model_pred_full - model_pred_pix
  output = lambda_pix * model_pred_pix + lambda_sem * delta
"""

import os
import sys
sys.path.append("/data/wyb/OSEDiff")
import time
import random
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules
from peft.utils.other import transpose

sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook

from osediff_vae_unet_loss_hybrid_wavelora import (
    inject_dual_wave_to_unet,
    add_lora_to_unet_attention,
)

import glob
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None

import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# ── initialize_unet ────────────────────────────────────────────────────────

def initialize_unet(rank_attn=4, pretrained_model_path=None,
                    wave_dim=None, wave_res=64, mlp_ratio=1.0, wave_scale=0.2,
                    return_module_names=False):
    """
    1. 注入双 WaveAdapter（conv 层，pix + sem）
    2. 加 LoRA（attention 层，pix + sem）
    顺序：WaveAdapter 注入必须在 add_adapter 之前
    """
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    # Step 1: 双 WaveAdapter 注入 conv 层
    pix_wave_modules, sem_wave_modules = inject_dual_wave_to_unet(
        unet, wave_dim=wave_dim, res=wave_res,
        mlp_ratio=mlp_ratio, scale=wave_scale)

    # Step 2: LoRA 注入 attention 层（与 WaveAdapter 正交）
    # 分 pix / sem 两套，和原版一样用 adapter_name 区分
    l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []

    attn_patterns = ["to_k", "to_q", "to_v", "to_out.0",
                     "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]

    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        # 跳过 WaveAdapter 参数
        if any(k in n for k in ["pix_adapter", "sem_adapter", "wave", "adapter"]):
            continue
        for pat in attn_patterns:
            if pat in n:
                mn = n.replace(".weight", "")
                if "down_blocks" in n or "conv_in" in n:
                    l_target_modules_encoder_pix.append(mn)
                    l_target_modules_encoder_sem.append(mn)
                elif "up_blocks" in n or "conv_out" in n:
                    l_target_modules_decoder_pix.append(mn)
                    l_target_modules_decoder_sem.append(mn)
                else:
                    l_modules_others_pix.append(mn)
                    l_modules_others_sem.append(mn)
                break

    unet.add_adapter(LoraConfig(r=rank_attn, init_lora_weights="gaussian",
                                target_modules=l_target_modules_encoder_pix),
                     adapter_name="default_encoder_pix")
    unet.add_adapter(LoraConfig(r=rank_attn, init_lora_weights="gaussian",
                                target_modules=l_target_modules_decoder_pix),
                     adapter_name="default_decoder_pix")
    unet.add_adapter(LoraConfig(r=rank_attn, init_lora_weights="gaussian",
                                target_modules=l_modules_others_pix),
                     adapter_name="default_others_pix")
    unet.add_adapter(LoraConfig(r=rank_attn, init_lora_weights="gaussian",
                                target_modules=l_target_modules_encoder_sem),
                     adapter_name="default_encoder_sem")
    unet.add_adapter(LoraConfig(r=rank_attn, init_lora_weights="gaussian",
                                target_modules=l_target_modules_decoder_sem),
                     adapter_name="default_decoder_sem")
    unet.add_adapter(LoraConfig(r=rank_attn, init_lora_weights="gaussian",
                                target_modules=l_modules_others_sem),
                     adapter_name="default_others_sem")

    if return_module_names:
        return (unet, pix_wave_modules, sem_wave_modules,
                l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix,
                l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem)
    return unet, pix_wave_modules, sem_wave_modules


# ── CSDLoss（原版完全不动）────────────────────────────────────────────────

class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(
            args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16
        self.unet_fix = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path_csd, subfolder="unet")
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available")
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        return model(latents, timestep=timestep,
                     encoder_hidden_states=prompt_embeds).sample

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        return (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    def cal_csd(self, latents, prompt_embeds, negative_prompt_embeds, args):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)
        timestep = torch.randint(min_dm_step, max_dm_step,
                                 (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)
        with torch.no_grad():
            noisy_cat = torch.cat([noisy_latents] * 2)
            t_cat     = torch.cat([timestep] * 2)
            pe_cat    = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_cat.to(dtype=torch.float16),
                timestep=t_cat,
                prompt_embeds=pe_cat.to(dtype=torch.float16))
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = (noise_pred_uncond
                              + args.cfg_csd * (noise_pred_text - noise_pred_uncond))
            pred_real = self.eps_to_mu(self.sched, noise_pred_cfg,
                                       noisy_latents, timestep)
            pred_fake = self.eps_to_mu(self.sched, noise_pred_uncond,
                                       noisy_latents, timestep)
        w = torch.abs(latents - pred_real).mean(dim=[1, 2, 3], keepdim=True)
        grad = (pred_fake - pred_real) / w
        return F.mse_loss(latents, (latents - grad).detach())

    def stopgrad(self, x): return x.detach()


# ── PiSASR_WaveLoRA（训练用）──────────────────────────────────────────────

class PiSASR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        wave_dim   = getattr(args, 'wave_dim',   None)
        wave_res   = getattr(args, 'wave_res',   64)
        mlp_ratio  = getattr(args, 'mlp_ratio',  1.0)
        wave_scale = getattr(args, 'wave_scale', 0.2)
        rank_attn  = getattr(args, 'lora_rank_unet_pix', 4)
        self._wave_cfg = dict(wave_dim=wave_dim, wave_res=wave_res,
                              mlp_ratio=mlp_ratio, wave_scale=wave_scale)
        self._wave_scale_train = wave_scale

        if args.resume_ckpt is None:
            (self.unet, self.pix_wave_modules, self.sem_wave_modules,
             self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix,
             self.lora_unet_others_pix,
             self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem,
             self.lora_unet_others_sem) = initialize_unet(
                rank_attn=rank_attn,
                pretrained_model_path=args.pretrained_model_path,
                wave_dim=wave_dim, wave_res=wave_res,
                mlp_ratio=mlp_ratio, wave_scale=wave_scale,
                return_module_names=True)
            self.lora_rank_unet_pix = rank_attn
            self.lora_rank_unet_sem = rank_attn
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(
                args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = SimpleNamespace(**read_yaml(stage1_yaml))
            self.unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_path, subfolder="unet")
            self.lora_rank_unet_pix = stage1_args.lora_rank_unet_pix
            self.lora_rank_unet_sem = stage1_args.lora_rank_unet_pix
            self.pix_wave_modules, self.sem_wave_modules = inject_dual_wave_to_unet(
                self.unet, wave_dim=wave_dim, wave_res=wave_res,
                mlp_ratio=mlp_ratio, scale=wave_scale)
            self.load_ckpt_from_state_dict(torch.load(args.resume_ckpt))

        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(
            args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')
        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()
        # 初始化时冻结 sem WaveAdapter
        self._freeze_sem_wave()

    # ── Wave 管理 ──────────────────────────────────────────────────────────

    def _freeze_sem_wave(self):
        for blk in self.sem_wave_modules.values():
            blk.requires_grad_(False)
            blk.scale = 0.0

    def _unfreeze_sem_wave(self):
        for blk in self.sem_wave_modules.values():
            blk.train()
            blk.requires_grad_(True)
            blk.scale = self._wave_scale_train

    def _freeze_pix_wave(self):
        for blk in self.pix_wave_modules.values():
            blk.requires_grad_(False)

    def _unfreeze_pix_wave(self):
        for blk in self.pix_wave_modules.values():
            blk.train()
            blk.requires_grad_(True)

    def get_pix_wave_params(self):
        params = []
        for blk in self.pix_wave_modules.values():
            params += list(blk.parameters())
        return params

    def get_sem_wave_params(self):
        params = []
        for blk in self.sem_wave_modules.values():
            params += list(blk.parameters())
        return params

    # ── 两阶段训练开关 ────────────────────────────────────────────────────

    def set_train_pix(self):
        """Stage1：pix WaveAdapter + pix attention LoRA 训练"""
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "pix" in n: _p.requires_grad = True
            if "sem" in n: _p.requires_grad = False
        self._unfreeze_pix_wave()
        self._freeze_sem_wave()  # sem WaveAdapter scale=0，不影响 Stage1

    def set_train_sem(self):
        """Stage2：sem WaveAdapter + sem attention LoRA 训练，pix 冻结"""
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n: _p.requires_grad = True
            if "pix" in n: _p.requires_grad = False
        self._freeze_pix_wave()
        self._unfreeze_sem_wave()

    # ── checkpoint ────────────────────────────────────────────────────────

    def load_ckpt_from_state_dict(self, sd):
        self.unet.add_adapter(LoraConfig(
            r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian",
            target_modules=sd["unet_lora_encoder_modules_pix"]),
            adapter_name="default_encoder_pix")
        self.unet.add_adapter(LoraConfig(
            r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian",
            target_modules=sd["unet_lora_decoder_modules_pix"]),
            adapter_name="default_decoder_pix")
        self.unet.add_adapter(LoraConfig(
            r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian",
            target_modules=sd["unet_lora_others_modules_pix"]),
            adapter_name="default_others_pix")
        self.unet.add_adapter(LoraConfig(
            r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian",
            target_modules=sd["unet_lora_encoder_modules_sem"]),
            adapter_name="default_encoder_sem")
        self.unet.add_adapter(LoraConfig(
            r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian",
            target_modules=sd["unet_lora_decoder_modules_sem"]),
            adapter_name="default_decoder_sem")
        self.unet.add_adapter(LoraConfig(
            r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian",
            target_modules=sd["unet_lora_others_modules_sem"]),
            adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_pix = sd["unet_lora_encoder_modules_pix"]
        self.lora_unet_modules_decoder_pix = sd["unet_lora_decoder_modules_pix"]
        self.lora_unet_others_pix          = sd["unet_lora_others_modules_pix"]
        self.lora_unet_modules_encoder_sem = sd["unet_lora_encoder_modules_sem"]
        self.lora_unet_modules_decoder_sem = sd["unet_lora_decoder_modules_sem"]
        self.lora_unet_others_sem          = sd["unet_lora_others_modules_sem"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])

        # 加载 WaveAdapter 权重
        if "state_dict_pix_wave" in sd:
            for key, blk in self.pix_wave_modules.items():
                blk_sd = {k[len(key)+1:]: v
                          for k, v in sd["state_dict_pix_wave"].items()
                          if k.startswith(key + ".")}
                if blk_sd: blk.load_state_dict(blk_sd, strict=True)
        if "state_dict_sem_wave" in sd:
            for key, blk in self.sem_wave_modules.items():
                blk_sd = {k[len(key)+1:]: v
                          for k, v in sd["state_dict_sem_wave"].items()
                          if k.startswith(key + ".")}
                if blk_sd: blk.load_state_dict(blk_sd, strict=True)

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            return torch.concat([
                self.text_encoder(
                    self.tokenizer(cap, max_length=self.tokenizer.model_max_length,
                                   padding="max_length", truncation=True,
                                   return_tensors="pt").input_ids.to(
                        self.text_encoder.device))[0]
                for cap in prompt_batch], dim=0)

    def forward(self, c_t, c_tgt, batch=None, args=None):
        encoded_control = (self.vae_fix.encode(c_t).latent_dist.sample()
                           * self.vae_fix.config.scaling_factor)
        prompt_embeds      = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds  = self.encode_prompt(batch["neg_prompt"])
        null_prompt_embeds = self.encode_prompt(batch["null_prompt"])
        pos_caption_enc = (null_prompt_embeds
                           if random.random() < args.null_text_ratio
                           else prompt_embeds)
        model_pred = self.unet(encoded_control, self.timesteps1,
                               encoder_hidden_states=pos_caption_enc.to(
                                   torch.float32)).sample
        x_denoised = encoded_control - model_pred
        output_image = (self.vae_fix.decode(
            x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)
        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_pix"] = self.lora_unet_modules_encoder_pix
        sd["unet_lora_decoder_modules_pix"] = self.lora_unet_modules_decoder_pix
        sd["unet_lora_others_modules_pix"]  = self.lora_unet_others_pix
        sd["unet_lora_encoder_modules_sem"] = self.lora_unet_modules_encoder_sem
        sd["unet_lora_decoder_modules_sem"] = self.lora_unet_modules_decoder_sem
        sd["unet_lora_others_modules_sem"]  = self.lora_unet_others_sem
        sd["lora_rank_unet_pix"] = self.lora_rank_unet_pix
        sd["lora_rank_unet_sem"] = self.lora_rank_unet_sem
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items()
                                 if "lora" in k}
        sd["wave_config"] = self._wave_cfg
        # 保存双 WaveAdapter 权重
        pix_sd = {}
        for k, blk in self.pix_wave_modules.items():
            for pn, pv in blk.state_dict().items():
                pix_sd[f"{k}.{pn}"] = pv
        sd["state_dict_pix_wave"] = pix_sd
        sem_sd = {}
        for k, blk in self.sem_wave_modules.items():
            for pn, pv in blk.state_dict().items():
                sem_sd[f"{k}.{pn}"] = pv
        sd["state_dict_sem_wave"] = sem_sd
        torch.save(sd, outf)


# ── PiSASR_eval（推理用）──────────────────────────────────────────────────

class PiSASR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device       = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args         = args

        self.tokenizer    = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched        = DDPMScheduler.from_pretrained(
            args.pretrained_model_path, subfolder="scheduler")
        self.vae          = AutoencoderKL.from_pretrained(
            args.pretrained_model_path, subfolder="vae")
        self.unet         = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path, subfolder="unet")

        self._load_pretrained_weights(args.pretrained_path)
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size)

        if not args.default:
            self._prepare_lora_deltas(
                ["default_encoder_sem", "default_decoder_sem", "default_others_sem"])
        set_weights_and_activate_adapters(
            self.unet,
            ["default_encoder_sem", "default_decoder_sem", "default_others_sem"],
            [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        # WaveAdapter 注入必须在 merge_and_unload 之后
        self._load_wave_after_merge()
        self._move_models_to_device_and_dtype()

        # merge 之后重建 ori_unet_weight（WaveAdapter 改变了 conv 名字）
        self.ori_unet_weight = {}
        for name, param in self.unet.named_parameters():
            self.ori_unet_weight[name] = param.clone().data.to(
                self.weight_dtype).to("cuda")

        self.timesteps1 = torch.tensor([1], device=self.device).long()
        self.lambda_pix = torch.tensor([args.lambda_pix], device=self.device)
        self.lambda_sem = torch.tensor([args.lambda_sem], device=self.device)

    def _load_wave_after_merge(self):
        """merge_and_unload 之后注入双 WaveAdapter 并加载权重"""
        if not hasattr(self, '_pending_wave_sd') or self._pending_wave_sd is None:
            self.pix_wave_modules = {}
            self.sem_wave_modules = {}
            return
        cfg = self._pending_wave_cfg
        self.pix_wave_modules, self.sem_wave_modules = inject_dual_wave_to_unet(
            self.unet,
            wave_dim=cfg.get("wave_dim"),
            res=cfg.get("wave_res", 64),
            mlp_ratio=cfg.get("mlp_ratio", 1.0),
            scale=cfg.get("wave_scale", 0.2))

        for key, blk in self.pix_wave_modules.items():
            blk_sd = {k[len(key)+1:]: v
                      for k, v in self._pending_wave_sd["pix"].items()
                      if k.startswith(key + ".")}
            if blk_sd: blk.load_state_dict(blk_sd, strict=True)
            blk.eval(); blk.requires_grad_(False)

        for key, blk in self.sem_wave_modules.items():
            blk_sd = {k[len(key)+1:]: v
                      for k, v in self._pending_wave_sd["sem"].items()
                      if k.startswith(key + ".")}
            if blk_sd: blk.load_state_dict(blk_sd, strict=True)
            blk.eval(); blk.requires_grad_(False)
            blk.scale = cfg.get("wave_scale", 0.2)  # 推理时 sem 也激活

        del self._pending_wave_sd, self._pending_wave_cfg

    def _get_dtype(self, precision):
        if precision == "fp16": return torch.float16
        elif precision == "bf16": return torch.bfloat16
        else: return torch.float32

    def _move_models_to_device_and_dtype(self):
        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        self._load_and_save_ckpt_from_state_dict(torch.load(pretrained_path))

    def _prepare_lora_deltas(self, adapter_names):
        self.lora_deltas_sem = {}
        for key, _ in self.unet.named_modules():
            if "lora_" in key: continue
            try: parent, target, target_name = _get_submodules(self.unet, key)
            except AttributeError: continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    for active_adapter in adapter_names:
                        if active_adapter in target.lora_A.keys():
                            weight_A = target.lora_A[active_adapter].weight
                            weight_B = target.lora_B[active_adapter].weight
                            s = target.get_base_layer().weight.size()
                            if s[2:4] == (1, 1):
                                out = (weight_B.squeeze(3).squeeze(2)
                                       @ weight_A.squeeze(3).squeeze(2)
                                       ).unsqueeze(2).unsqueeze(3) * target.scaling[active_adapter]
                            elif len(s) == 2:
                                out = transpose(weight_B @ weight_A, False) * target.scaling[active_adapter]
                            else:
                                out = F.conv2d(weight_A.permute(1, 0, 2, 3),
                                               weight_B).permute(1, 0, 2, 3) * target.scaling[active_adapter]
                            self.lora_deltas_sem[key + ".weight"] = out.data.to(
                                dtype=self.weight_dtype, device=self.device)

    def _apply_lora_delta(self):
        for name, param in self.unet.named_parameters():
            if name in self.lora_deltas_sem:
                param.data = self.lora_deltas_sem[name] + self.ori_unet_weight[name]
            else:
                param.data = self.ori_unet_weight[name]

    def _apply_ori_weight(self):
        for name, param in self.unet.named_parameters():
            param.data = self.ori_unet_weight[name]

    def _apply_sem_wave_scale(self, scale):
        for blk in self.sem_wave_modules.values():
            blk.scale = scale

    def _load_and_save_ckpt_from_state_dict(self, sd):
        lc_ep = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_encoder_modules_pix"])
        lc_dp = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_decoder_modules_pix"])
        lc_op = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_others_modules_pix"])
        lc_es = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_encoder_modules_sem"])
        lc_ds = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_decoder_modules_sem"])
        lc_os = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian",
                           target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(lc_ep, adapter_name="default_encoder_pix")
        self.unet.add_adapter(lc_dp, adapter_name="default_decoder_pix")
        self.unet.add_adapter(lc_op, adapter_name="default_others_pix")
        for name, param in self.unet.named_parameters():
            if "pix" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        set_weights_and_activate_adapters(
            self.unet,
            ["default_encoder_pix", "default_decoder_pix", "default_others_pix"],
            [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()
        self.ori_unet_weight = {}
        for name, param in self.unet.named_parameters():
            self.ori_unet_weight[name] = param.clone().data.to(
                self.weight_dtype).to("cuda")

        self.unet.add_adapter(lc_es, adapter_name="default_encoder_sem")
        self.unet.add_adapter(lc_ds, adapter_name="default_decoder_sem")
        self.unet.add_adapter(lc_os, adapter_name="default_others_sem")
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        if "state_dict_pix_wave" in sd and "state_dict_sem_wave" in sd:
            self._pending_wave_sd  = {"pix": sd["state_dict_pix_wave"],
                                      "sem": sd["state_dict_sem_wave"]}
            self._pending_wave_cfg = sd.get("wave_config", {})
        else:
            self._pending_wave_sd  = None
            self._pending_wave_cfg = {}

    def set_eval(self):
        self.unet.eval(); self.vae.eval()
        self.unet.requires_grad_(False); self.vae.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            return torch.concat([
                self.text_encoder(
                    self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                                   padding="max_length", truncation=True,
                                   return_tensors="pt").input_ids.to(
                        self.text_encoder.device))[0]
                for caption in prompt_batch], dim=0)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters()) / 1e9

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        torch.cuda.synchronize()
        start_time = time.time()
        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds   = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        encoded_control = (self.vae.encode(c_t).latent_dist.sample()
                           * self.vae.config.scaling_factor)
        model_pred = self._process_latents(encoded_control, prompt_embeds, default)
        x_denoised   = encoded_control - model_pred
        output_image = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        torch.cuda.synchronize()
        return time.time() - start_time, output_image

    def _process_latents(self, encoded_control, prompt_embeds, default):
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = (self.args.latent_tiled_size,
                                   self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            return self._predict_no_tiling(encoded_control, prompt_embeds, default)
        return self._predict_with_tiling(
            encoded_control, prompt_embeds, default, tile_size, tile_overlap)

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default):
        if default:
            return self.unet(encoded_control, self.timesteps1,
                             encoder_hidden_states=prompt_embeds).sample
        # 第一次前向：pix + sem 都激活
        model_pred_sem = self.unet(encoded_control, self.timesteps1,
                                   encoder_hidden_states=prompt_embeds).sample
        # 关闭 sem：LoRA 用 ori_weight，WaveAdapter 用 scale=0
        self._apply_ori_weight()
        self._apply_sem_wave_scale(0.0)
        model_pred_pix = self.unet(encoded_control, self.timesteps1,
                                   encoder_hidden_states=prompt_embeds).sample
        # 恢复
        self._apply_lora_delta()
        self._apply_sem_wave_scale(self.args.wave_scale
                                   if hasattr(self.args, 'wave_scale') else 0.2)
        model_pred_sem -= model_pred_pix
        return self.lambda_pix * model_pred_pix + self.lambda_sem * model_pred_sem

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default,
                              tile_size, tile_overlap):
        _, _, h, w = encoded_control.size()
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        grid_rows = 0; cur_x = 0
        while cur_x < encoded_control.size(-1):
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1
        grid_cols = 0; cur_y = 0
        while cur_y < encoded_control.size(-2):
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        input_list = []; noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1: ofs_x = w - tile_size
                if col == grid_cols - 1: ofs_y = h - tile_size
                input_list.append(
                    encoded_control[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size])
                if len(input_list) == 1 or col == grid_cols - 1:
                    t = torch.cat(input_list, dim=0)
                    if default:
                        model_out = self.unet(t, self.timesteps1,
                                             encoder_hidden_states=prompt_embeds).sample
                    else:
                        model_out_sem = self.unet(t, self.timesteps1,
                                                  encoder_hidden_states=prompt_embeds).sample
                        self._apply_ori_weight()
                        self._apply_sem_wave_scale(0.0)
                        model_out_pix = self.unet(t, self.timesteps1,
                                                  encoder_hidden_states=prompt_embeds).sample
                        self._apply_lora_delta()
                        self._apply_sem_wave_scale(
                            self.args.wave_scale if hasattr(self.args, 'wave_scale') else 0.2)
                        model_out_sem -= model_out_pix
                        model_out = (self.lambda_pix * model_out_pix
                                     + self.lambda_sem * model_out_sem)
                    input_list = []
                noise_preds.append(model_out)

        noise_pred   = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        idx = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1: ofs_x = w - tile_size
                if col == grid_cols - 1: ofs_y = h - tile_size
                noise_pred[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size] += (
                    noise_preds[idx] * tile_weights)
                contributors[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size] += tile_weights
                idx += 1
        return noise_pred / contributors

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        from numpy import pi, exp, sqrt
        import numpy as np
        mx = (tile_width - 1) / 2; my = (tile_height - 1) / 2
        xp = [exp(-(x-mx)**2/(2*(tile_width**2)*0.01))/sqrt(2*pi*0.01)
              for x in range(tile_width)]
        yp = [exp(-(y-my)**2/(2*(tile_height**2)*0.01))/sqrt(2*pi*0.01)
              for y in range(tile_height)]
        w = np.outer(yp, xp)
        return torch.tensor(w, device=self.device).repeat(
            nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256,
                        fast_decoder=False, fast_encoder=False,
                        color_fix=False, vae_to_gpu=True):
        enc, dec = self.vae.encoder, self.vae.decoder
        if not hasattr(enc, 'original_forward'): enc.original_forward = enc.forward
        if not hasattr(dec, 'original_forward'): dec.original_forward = dec.forward
        enc.forward = VAEHook(enc, encoder_tile_size, is_decoder=False,
                              fast_decoder=fast_decoder, fast_encoder=fast_encoder,
                              color_fix=color_fix, to_gpu=vae_to_gpu)
        dec.forward = VAEHook(dec, decoder_tile_size, is_decoder=True,
                              fast_decoder=fast_decoder, fast_encoder=fast_encoder,
                              color_fix=color_fix, to_gpu=vae_to_gpu)