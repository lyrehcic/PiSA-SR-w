"""
pisasr_HF.py

在原版 PiSASR 基础上加入 HDW-SR 启发的高频子带指导：
  - Stage1（pix LoRA）：完全不变
  - Stage2（sem LoRA）：用 LR latent 的 DWT 高频子带（LH/HL/HH）
    投影成 8 个额外 token，concat 到 text embedding 后面，
    给 sem LoRA 的 cross-attention 提供显式高频先验

参考论文：
  HDW-SR (arXiv:2511.13175) - High-Frequency Guided Diffusion Model
  based on Wavelet Decomposition for Image Super-Resolution
"""

import os
import sys
import time
import random
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules, ModulesToSaveWrapper
from peft.utils.other import transpose
import pywt

sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook


import glob
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# ── DWT_2D（来自 DiMSUM，经过验证的实现）────────────────────────────────────

class DWT_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        self.register_buffer("w_ll", w_ll.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer("w_lh", w_lh.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer("w_hl", w_hl.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer("w_hh", w_hh.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


# ── HFGuidanceProj：把高频子带投影成 cross-attention token ─────────────────

class HFGuidanceProj(nn.Module):
    """
    输入：LR latent 的 DWT 高频子带 [B, 12, H/2, W/2]
          (4通道latent × 3高频子带LH/HL/HH = 12通道)
    输出：[B, num_tokens, 1024] 的 cross-attention token
    零初始化：训练初期输出全零，不干扰原有 text condition
    """
    def __init__(self, in_channels=12, num_tokens=8, token_dim=1024, pool_size=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        flat_dim = in_channels * pool_size * pool_size

        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_tokens * token_dim),
        )
        # 零初始化最后一层，Stage2 初始完全不干扰
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, hf_bands):
        # hf_bands: [B, 12, H/2, W/2]
        B = hf_bands.shape[0]
        x = self.pool(hf_bands)          # [B, 12, pool_size, pool_size]
        x = x.flatten(1)                  # [B, 12*pool_size*pool_size]
        x = self.proj(x)                  # [B, num_tokens * token_dim]
        x = x.reshape(B, self.num_tokens, self.token_dim)  # [B, num_tokens, 1024]
        return x


# ── initialize_unet（原版不动）────────────────────────────────────────────

def initialize_unet(rank_pix, rank_sem, return_lora_module_names=False, pretrained_model_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in",
              "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder_pix.append(n.replace(".weight", ""))
                l_target_modules_encoder_sem.append(n.replace(".weight", ""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder_pix.append(n.replace(".weight", ""))
                l_target_modules_decoder_sem.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others_pix.append(n.replace(".weight", ""))
                l_modules_others_sem.append(n.replace(".weight", ""))
                break

    unet.add_adapter(LoraConfig(r=rank_pix, init_lora_weights="gaussian", target_modules=l_target_modules_encoder_pix), adapter_name="default_encoder_pix")
    unet.add_adapter(LoraConfig(r=rank_pix, init_lora_weights="gaussian", target_modules=l_target_modules_decoder_pix), adapter_name="default_decoder_pix")
    unet.add_adapter(LoraConfig(r=rank_pix, init_lora_weights="gaussian", target_modules=l_modules_others_pix),          adapter_name="default_others_pix")
    unet.add_adapter(LoraConfig(r=rank_sem, init_lora_weights="gaussian", target_modules=l_target_modules_encoder_sem),  adapter_name="default_encoder_sem")
    unet.add_adapter(LoraConfig(r=rank_sem, init_lora_weights="gaussian", target_modules=l_target_modules_decoder_sem),  adapter_name="default_decoder_sem")
    unet.add_adapter(LoraConfig(r=rank_sem, init_lora_weights="gaussian", target_modules=l_modules_others_sem),           adapter_name="default_others_sem")

    if return_lora_module_names:
        return (unet,
                l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix,
                l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem)
    return unet


# ── CSDLoss（原版完全不动）────────────────────────────────────────────────

class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available")
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        return model(latents, timestep=timestep, encoder_hidden_states=prompt_embeds).sample

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        return (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    def cal_csd(self, latents, prompt_embeds, negative_prompt_embeds, args):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)
        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)
        with torch.no_grad():
            noisy_cat  = torch.cat([noisy_latents] * 2)
            t_cat      = torch.cat([timestep] * 2)
            pe_cat     = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_cat.to(dtype=torch.float16),
                timestep=t_cat,
                prompt_embeds=pe_cat.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            pred_real = self.eps_to_mu(self.sched, noise_pred_cfg,    noisy_latents, timestep)
            pred_fake = self.eps_to_mu(self.sched, noise_pred_uncond,  noisy_latents, timestep)
        w = torch.abs(latents - pred_real).mean(dim=[1, 2, 3], keepdim=True)
        grad = (pred_fake - pred_real) / w
        return F.mse_loss(latents, (latents - grad).detach())

    def stopgrad(self, x):
        return x.detach()


# ── PiSASR_HF（训练用）───────────────────────────────────────────────────

class PiSASR(torch.nn.Module):
    """
    在原版 PiSASR 基础上加入高频子带指导。
    训练脚本 import PiSASR 时直接用本文件即可，接口完全兼容。
    """
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        if args.resume_ckpt is None:
            (self.unet,
             self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix,
             self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem) = \
                initialize_unet(rank_pix=args.lora_rank_unet_pix, rank_sem=args.lora_rank_unet_sem,
                                pretrained_model_path=args.pretrained_model_path,
                                return_lora_module_names=True)
            self.lora_rank_unet_pix = args.lora_rank_unet_pix
            self.lora_rank_unet_sem = args.lora_rank_unet_sem
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = SimpleNamespace(**read_yaml(stage1_yaml))
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
            self.lora_rank_unet_pix = stage1_args.lora_rank_unet_pix
            self.lora_rank_unet_sem = stage1_args.lora_rank_unet_sem
            self.load_ckpt_from_state_dict(torch.load(args.resume_ckpt))

        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')

        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()

        # ── 高频指导模块 ──────────────────────────────────────────────────
        # DWT：不可训练，只用于提取特征
        self.dwt = DWT_2D(wave='haar').cuda()
        self.dwt.requires_grad_(False)

        # HF proj：可训练，零初始化，Stage2 才激活梯度
        # latent 4通道 × 高频3子带 = 12通道输入
        self.hf_proj = HFGuidanceProj(
            in_channels=12,
            num_tokens=8,
            token_dim=1024,
            pool_size=4,
        ).cuda()

        # Stage 标志
        self._in_sem_stage = False
        # ────────────────────────────────────────────────────────────────

    def set_train_pix(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "pix" in n: _p.requires_grad = True
            if "sem" in n: _p.requires_grad = False
        # Stage1 不训练 hf_proj
        self.hf_proj.requires_grad_(False)
        self._in_sem_stage = False

    def set_train_sem(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n: _p.requires_grad = True
            if "pix" in n: _p.requires_grad = False
        # Stage2 激活 hf_proj 梯度
        self.hf_proj.requires_grad_(True)
        self.hf_proj.train()
        self._in_sem_stage = True

    def load_ckpt_from_state_dict(self, sd):
        self.unet.add_adapter(LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"]), adapter_name="default_encoder_pix")
        self.unet.add_adapter(LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"]), adapter_name="default_decoder_pix")
        self.unet.add_adapter(LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"]),  adapter_name="default_others_pix")
        self.unet.add_adapter(LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"]), adapter_name="default_encoder_sem")
        self.unet.add_adapter(LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"]), adapter_name="default_decoder_sem")
        self.unet.add_adapter(LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"]),  adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_pix = sd["unet_lora_encoder_modules_pix"]
        self.lora_unet_modules_decoder_pix = sd["unet_lora_decoder_modules_pix"]
        self.lora_unet_others_pix          = sd["unet_lora_others_modules_pix"]
        self.lora_unet_modules_encoder_sem = sd["unet_lora_encoder_modules_sem"]
        self.lora_unet_modules_decoder_sem = sd["unet_lora_decoder_modules_sem"]
        self.lora_unet_others_sem          = sd["unet_lora_others_modules_sem"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])

        # 如果 checkpoint 里有 hf_proj 权重就加载，没有就保持零初始化
        if "state_dict_hf_proj" in sd:
            self.hf_proj.load_state_dict(sd["state_dict_hf_proj"])

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            return torch.concat([
                self.text_encoder(
                    self.tokenizer(cap, max_length=self.tokenizer.model_max_length,
                                   padding="max_length", truncation=True,
                                   return_tensors="pt").input_ids.to(self.text_encoder.device)
                )[0]
                for cap in prompt_batch
            ], dim=0)

    def forward(self, c_t, c_tgt, batch=None, args=None):
        encoded_control   = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        prompt_embeds     = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        null_prompt_embeds= self.encode_prompt(batch["null_prompt"])

        if random.random() < args.null_text_ratio:
            pos_caption_enc = null_prompt_embeds
        else:
            pos_caption_enc = prompt_embeds

        # ── Stage2：注入高频子带 token ───────────────────────────────────
        if self._in_sem_stage:
            with torch.no_grad():
                wav = self.dwt(encoded_control.float())          # [B, 16, H/2, W/2]
                B, C4, H2, W2 = wav.shape
                C = C4 // 4
                wav_bands = wav.reshape(B, 4, C, H2, W2)
                # 取 LH/HL/HH 三个高频子带，index 1,2,3
                hf_bands = wav_bands[:, 1:, :, :, :].reshape(B, 3 * C, H2, W2)  # [B, 12, H/2, W/2]

            hf_tokens = self.hf_proj(hf_bands.float())          # [B, 8, 1024]
            cond = torch.cat([pos_caption_enc.float(), hf_tokens], dim=1)  # [B, 85, 1024]
        else:
            cond = pos_caption_enc.float()
        # ────────────────────────────────────────────────────────────────

        model_pred   = self.unet(encoded_control, self.timesteps1,
                                 encoder_hidden_states=cond).sample
        x_denoised   = encoded_control - model_pred
        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_pix"] = self.lora_unet_modules_encoder_pix
        sd["unet_lora_decoder_modules_pix"] = self.lora_unet_modules_decoder_pix
        sd["unet_lora_others_modules_pix"]  = self.lora_unet_others_pix
        sd["unet_lora_encoder_modules_sem"] = self.lora_unet_modules_encoder_sem
        sd["unet_lora_decoder_modules_sem"] = self.lora_unet_modules_decoder_sem
        sd["unet_lora_others_modules_sem"]  = self.lora_unet_others_sem
        sd["lora_rank_unet_pix"]            = self.lora_rank_unet_pix
        sd["lora_rank_unet_sem"]            = self.lora_rank_unet_sem
        sd["state_dict_unet"]  = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        sd["state_dict_hf_proj"] = self.hf_proj.state_dict()
        torch.save(sd, outf)


# ── PiSASR_eval（推理用，原版不动）────────────────────────────────────────

class PiSASR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args

        self.tokenizer    = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched        = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet         = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        self._load_pretrained_weights(args.pretrained_path)
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size,
                             decoder_tile_size=args.vae_decoder_tiled_size)

        if not args.default:
            self._prepare_lora_deltas(["default_encoder_sem", "default_decoder_sem", "default_others_sem"])
        set_weights_and_activate_adapters(self.unet,
            ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()
        self._move_models_to_device_and_dtype()

        self.timesteps1  = torch.tensor([1], device=self.device).long()
        self.lambda_pix  = torch.tensor([args.lambda_pix], device=self.device)
        self.lambda_sem  = torch.tensor([args.lambda_sem], device=self.device)

        # 推理时也支持高频指导（如果 checkpoint 里有 hf_proj）
        self.hf_proj = None
        if hasattr(self, '_pending_hf_proj_sd') and self._pending_hf_proj_sd is not None:
            self.dwt = DWT_2D(wave='haar').to(self.device, dtype=self.weight_dtype)
            self.dwt.requires_grad_(False)
            self.hf_proj = HFGuidanceProj(in_channels=12, num_tokens=8,
                                          token_dim=1024, pool_size=4).to(self.device, dtype=self.weight_dtype)
            self.hf_proj.load_state_dict(self._pending_hf_proj_sd)
            self.hf_proj.eval()
            self.hf_proj.requires_grad_(False)
            del self._pending_hf_proj_sd

    def _get_dtype(self, precision):
        if precision == "fp16":   return torch.float16
        elif precision == "bf16": return torch.bfloat16
        else:                     return torch.float32

    def _move_models_to_device_and_dtype(self):
        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        sd = torch.load(pretrained_path)
        self._load_and_save_ckpt_from_state_dict(sd)

    def _prepare_lora_deltas(self, adapter_names):
        self.lora_deltas_sem = {}
        key_list = [key for key, _ in self.unet.named_modules() if "lora_" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.unet, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    for active_adapter in adapter_names:
                        if active_adapter in target.lora_A.keys():
                            weight_A = target.lora_A[active_adapter].weight
                            weight_B = target.lora_B[active_adapter].weight
                            s = target.get_base_layer().weight.size()
                            if s[2:4] == (1, 1):
                                output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * target.scaling[active_adapter]
                            elif len(s) == 2:
                                output_tensor = transpose(weight_B @ weight_A, False) * target.scaling[active_adapter]
                            else:
                                output_tensor = F.conv2d(weight_A.permute(1, 0, 2, 3), weight_B).permute(1, 0, 2, 3) * target.scaling[active_adapter]
                            self.lora_deltas_sem[key + ".weight"] = output_tensor.data.to(dtype=self.weight_dtype, device=self.device)

    def _apply_lora_delta(self):
        for name, param in self.unet.named_parameters():
            if name in self.lora_deltas_sem:
                param.data = self.lora_deltas_sem[name] + self.ori_unet_weight[name]
            else:
                param.data = self.ori_unet_weight[name]

    def _apply_ori_weight(self):
        for name, param in self.unet.named_parameters():
            param.data = self.ori_unet_weight[name]

    def _load_and_save_ckpt_from_state_dict(self, sd):
        lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        lora_conf_others_pix  = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])
        lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        lora_conf_others_sem  = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(lora_conf_others_pix,  adapter_name="default_others_pix")

        for name, param in self.unet.named_parameters():
            if "pix" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        set_weights_and_activate_adapters(self.unet,
            ["default_encoder_pix", "default_decoder_pix", "default_others_pix"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()
        self.ori_unet_weight = {}
        for name, param in self.unet.named_parameters():
            self.ori_unet_weight[name] = param.clone().data.to(self.weight_dtype).to("cuda")

        self.unet.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(lora_conf_others_sem,  adapter_name="default_others_sem")
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        self._pending_hf_proj_sd = sd.get("state_dict_hf_proj", None)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            return torch.concat([
                self.text_encoder(
                    self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                                   padding="max_length", truncation=True,
                                   return_tensors="pt").input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ], dim=0)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters()) / 1e9

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds   = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        model_pred = self._process_latents(encoded_control, prompt_embeds, default)
        x_denoised   = encoded_control - model_pred
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        torch.cuda.synchronize()
        return time.time() - start_time, output_image

    def _process_latents(self, encoded_control, prompt_embeds, default):
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = self.args.latent_tiled_size, self.args.latent_tiled_overlap
        if h * w <= tile_size * tile_size:
            return self._predict_no_tiling(encoded_control, prompt_embeds, default)
        return self._predict_with_tiling(encoded_control, prompt_embeds, default, tile_size, tile_overlap)

    def _get_cond(self, encoded_control, prompt_embeds):
        """如果有 hf_proj，注入高频 token；否则直接用 text embedding。"""
        if self.hf_proj is not None:
            wav = self.dwt(encoded_control.float())
            B, C4, H2, W2 = wav.shape
            C = C4 // 4
            wav_bands = wav.reshape(B, 4, C, H2, W2)
            hf_bands  = wav_bands[:, 1:, :, :, :].reshape(B, 3 * C, H2, W2)
            hf_tokens = self.hf_proj(hf_bands.to(self.weight_dtype))
            return torch.cat([prompt_embeds, hf_tokens], dim=1)
        return prompt_embeds

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default):
        cond = self._get_cond(encoded_control, prompt_embeds)
        if default:
            return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=cond).sample
        model_pred_sem = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=cond).sample
        self._apply_ori_weight()
        model_pred_pix = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        self._apply_lora_delta()
        model_pred_sem -= model_pred_pix
        return self.lambda_pix * model_pred_pix + self.lambda_sem * model_pred_sem

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default, tile_size, tile_overlap):
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
                input_tile = encoded_control[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size]
                input_list.append(input_tile)
                if len(input_list) == 1 or col == grid_cols - 1:
                    input_list_t = torch.cat(input_list, dim=0)
                    cond = self._get_cond(input_list_t, prompt_embeds)
                    if default:
                        model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=cond).sample
                    else:
                        model_out_sem = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=cond).sample
                        self._apply_ori_weight()
                        model_out_pix = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
                        self._apply_lora_delta()
                        model_out_sem -= model_out_pix
                        model_out = self.lambda_pix * model_out_pix + self.lambda_sem * model_out_sem
                    input_list = []
                noise_preds.append(model_out)

        noise_pred   = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1: ofs_x = w - tile_size
                if col == grid_cols - 1: ofs_y = h - tile_size
                noise_pred[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size] += noise_preds[row * grid_cols + col] * tile_weights
                contributors[:, :, ofs_y:ofs_y+tile_size, ofs_x:ofs_x+tile_size] += tile_weights
        return noise_pred / contributors

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        from numpy import pi, exp, sqrt
        import numpy as np
        midpoint_x = (tile_width  - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [exp(-(x - midpoint_x) ** 2 / (2 * (tile_width  ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for x in range(tile_width)]
        y_probs = [exp(-(y - midpoint_y) ** 2 / (2 * (tile_height ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for y in range(tile_height)]
        weights = np.outer(y_probs, x_probs)
        return torch.tensor(weights, device=self.device).repeat(nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256,
                        fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        encoder, decoder = self.vae.encoder, self.vae.decoder
        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward
        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False,
                                  fast_decoder=fast_decoder, fast_encoder=fast_encoder,
                                  color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True,
                                  fast_decoder=fast_decoder, fast_encoder=fast_encoder,
                                  color_fix=color_fix, to_gpu=vae_to_gpu)