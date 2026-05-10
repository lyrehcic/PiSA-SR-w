
"""  
  修改记录 (2026-04-10):
    - Wave2D_Fixed 加入 freq_gain 参数
    - freq_gain shape: [res, res, 1]，初始化为低频=1，高频略>1
    - forward 里在波动方程之后、IDCT之前乘以 freq_gain
    - WaveAdapter 加入 freeze_freq_gain / unfreeze_freq_gain 方法
    - PiSASR.set_train_pix / set_train_sem 分别控制 freq_gain 的冻结
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from peft import LoraConfig


#频率freq（高低频不同）

def _make_freq_gain_init(res: int) -> torch.Tensor:

    import numpy as np
    fh = np.arange(res, dtype=np.float32).reshape(-1, 1)
    fw = np.arange(res, dtype=np.float32).reshape(1, -1)
    #到（0，0）的距离
    freq_dist = np.sqrt(fh**2 + fw**2) / (res * np.sqrt(2))
    # 初始增益：低频=1.0，高频=1.1
    gain = 1.0 + 0.1 * freq_dist
    return torch.tensor(gain, dtype=torch.float32).unsqueeze(-1)  # [res, res, 1]
    #wave_res影响


class Wave2D_Fixed(nn.Module):
    def __init__(self, dim: int, res: int = 64):
        super().__init__()
        self.dim        = dim
        self.res        = res
        self.linear     = nn.Linear(dim, 2 * dim, bias=True)
        self.gate_proj  = nn.Linear(dim, dim, bias=True)
        self.out_norm   = nn.LayerNorm(dim)
        self.out_linear = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.GELU(),
        )
        self.c     = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)


        # Stage1冻结，Stage2 sem解冻）
        self.freq_gain = nn.Parameter(
            _make_freq_gain_init(res),  # [res, res, 1]
            requires_grad=False  
        )

    @staticmethod
    def _make_cos_map(N: int, device, dtype=torch.float32):
        k = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N
        n = torch.arange(N, device=device, dtype=dtype)
        W = torch.cos(torch.outer(n, k) * math.pi) * math.sqrt(2.0 / N)
        W[0, :] /= math.sqrt(2)
        return W

    def _get_cos_maps(self, H, W, device):
        key = (H, W, device.type, getattr(device, 'index', 0))
        if getattr(self, '_cos_key', None) != key:
            self._cos_key = key
            self._cosH = self._make_cos_map(H, device).detach()
            self._cosW = self._make_cos_map(W, device).detach()
        return self._cosH, self._cosW

    @staticmethod
    def _dct2d(x, cosH, cosW):
        x = torch.einsum('bhwc,hf->bfwc', x, cosH)
        x = torch.einsum('bfwc,wg->bfgc', x, cosW)
        return x

    @staticmethod
    def _idct2d(x, cosH, cosW):
        x = torch.einsum('bfgc,wg->bfwc', x, cosW)
        x = torch.einsum('bfwc,hf->bhwc', x, cosH)
        return x

    def _get_freq_gain(self, H, W):
        """
        把 freq_gain 插值到当前 feature map 的 HW 尺寸。
        freq_gain 定义在 [res, res]，不同层的 HW 不同，需要插值。
        """
        fg = self.freq_gain 
        if (H, W) == (fg.shape[0], fg.shape[1]):
            return fg  

        fg_4d = fg.permute(2, 0, 1).unsqueeze(0).float()  # [1,1,res,res]
        fg_interp = F.interpolate(
            fg_4d, size=(H, W), mode='bilinear', align_corners=False
        )  # [1,1,H,W]
        return fg_interp.squeeze(0).permute(1, 2, 0).to(fg.dtype)  # [H,W,1]

    def forward(self, x: torch.Tensor, freq_embed=None):
        orig_dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1).contiguous()
        xz = self.linear.float()(x_cl)
        x_disp, z_vel = xz.chunk(2, dim=-1)
        v0 = F.silu(z_vel)
        cosH, cosW = self._get_cos_maps(H, W, x.device)
        u0_hat = self._dct2d(x_disp, cosH, cosW)
        v0_hat = self._dct2d(v0, cosH, cosW)
        u0_hat = torch.clamp(u0_hat, -100.0, 100.0)
        v0_hat = torch.clamp(v0_hat, -100.0, 100.0)
        if freq_embed is not None:
            fe = freq_embed.unsqueeze(0).expand(B, -1, -1, -1).float()
            t  = self.to_k.float()(fe)
        else:
            t = torch.zeros(B, H, W, self.dim, device=x.device, dtype=torch.float32)
        c_safe  = torch.abs(self.c.float()) + 1e-4
        alpha_s = torch.clamp(self.alpha.float(), min=0.0)
        ct      = torch.clamp(c_safe * t, -20.0, 20.0)


        u_hat = (torch.cos(ct) * u0_hat
                 + torch.sin(ct) / c_safe * (v0_hat + alpha_s / 2.0 * u0_hat))


        # u_hat shape: [B, H, W, C]
        # freq_gain shape: [H, W, 1] → 广播到 [B, H, W, C]
        gain = self._get_freq_gain(H, W).float()  # [H, W, 1]
        u_hat = u_hat * gain

        x_out = self._idct2d(u_hat, cosH, cosW)
        x_out = torch.clamp(x_out, -100.0, 100.0)
        x_out = self.out_norm.float()(x_out)
        gate  = F.silu(self.gate_proj.float()(x_disp))
        x_out = x_out * gate
        x_out = self.out_linear.float()(x_out)
        x_out = torch.nan_to_num(x_out, nan=0.0)
        return x_out.permute(0, 3, 1, 2).contiguous().to(orig_dtype)


# ── WaveAdapter ────────────────────────────────

class WaveAdapter(nn.Module):

    def __init__(self, channels: int, wave_dim: int = None,
                 res: int = 64, mlp_ratio: float = 1.0, scale: float = 0.2):
        super().__init__()
        self.channels = channels
        self.scale    = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)
        wave_dim      = wave_dim or channels

        self.norm_in  = nn.GroupNorm(min(32, channels), channels, eps=1e-6)
        self.proj_in  = (nn.Conv2d(channels, wave_dim, 1, bias=False)
                         if wave_dim != channels else nn.Identity())

        self.wave     = Wave2D_Fixed(dim=wave_dim, res=res)
        self.proj_out = nn.Conv2d(wave_dim, channels, 1, bias=True)

        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

        self.freq_embed = nn.Parameter(torch.zeros(res, res, wave_dim))
        trunc_normal_(self.freq_embed, std=0.02)

        # 零初始化
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def freeze_freq_gain(self):
        self.wave.freq_gain.requires_grad = False

    def unfreeze_freq_gain(self):
        self.wave.freq_gain.requires_grad = True

    def _get_freq(self, H, W):
        fe = self.freq_embed
        if (H, W) == (fe.shape[0], fe.shape[1]):
            return fe
        fe_4d = fe.permute(2, 0, 1).unsqueeze(0).float()
        fe_interp = F.interpolate(fe_4d, size=(H, W), mode='bilinear', align_corners=False)
        return fe_interp.squeeze(0).permute(1, 2, 0).to(fe.dtype)

    def forward(self, x: torch.Tensor):
        _, _, H, W = x.shape
        x_norm = self.norm_in(x)
        r = self.proj_in(x_norm)
        r = self.wave(r, self._get_freq(H, W))
        r = self.proj_out(r)
        f = self.ffn(x_norm)
        return (r + f) * self.scale



class _DualWaveConv(nn.Module):
    def __init__(self, orig_conv, pix_adapter, sem_adapter):
        super().__init__()
        self.conv        = orig_conv
        self.pix_adapter = pix_adapter
        self.sem_adapter = sem_adapter

    def forward(self, x, scale=None):
        h = self.conv(x)
        return h + self.pix_adapter(h) + self.sem_adapter(h)


def inject_dual_wave_to_unet(unet, wave_dim=None, res=64,
                              mlp_ratio=1.0, scale=0.2) -> tuple:
    pix_modules = {}
    sem_modules = {}

    def _blk(out_ch):
        wd = wave_dim or min(out_ch, 512)
        pix = WaveAdapter(channels=out_ch, wave_dim=wd,
                          res=res, mlp_ratio=mlp_ratio, scale=scale)
        sem = WaveAdapter(channels=out_ch, wave_dim=wd,
                          res=res, mlp_ratio=mlp_ratio, scale=0.0)
        return pix, sem

    def _try_wrap(parent, attr, tag):
        layer = getattr(parent, attr, None)
        if isinstance(layer, nn.Conv2d):
            pix, sem = _blk(layer.out_channels)
            setattr(parent, attr, _DualWaveConv(layer, pix, sem))
            pix_modules[tag] = pix
            sem_modules[tag] = sem

    _try_wrap(unet, 'conv_in',  'unet.conv_in')
    _try_wrap(unet, 'conv_out', 'unet.conv_out')

    for bi, block in enumerate(unet.down_blocks):
        for ri, rb in enumerate(getattr(block, 'resnets', [])):
            for cn in ('conv1', 'conv2', 'conv_shortcut'):
                _try_wrap(rb, cn, f'unet.down{bi}.res{ri}.{cn}')
        for di, ds in enumerate(getattr(block, 'downsamplers', []) or []):
            _try_wrap(ds, 'conv', f'unet.down{bi}.ds{di}.conv')

    for ri, rb in enumerate(getattr(unet.mid_block, 'resnets', [])):
        for cn in ('conv1', 'conv2', 'conv_shortcut'):
            _try_wrap(rb, cn, f'unet.mid.res{ri}.{cn}')

    for bi, block in enumerate(unet.up_blocks):
        for ri, rb in enumerate(getattr(block, 'resnets', [])):
            for cn in ('conv1', 'conv2', 'conv_shortcut'):
                _try_wrap(rb, cn, f'unet.up{bi}.res{ri}.{cn}')
        for ui, up in enumerate(getattr(block, 'upsamplers', []) or []):
            _try_wrap(up, 'conv', f'unet.up{bi}.us{ui}.conv')

    return pix_modules, sem_modules


def add_lora_to_unet_attention(unet, lora_rank=4) -> list:
    l_attn = []
    attn_patterns = [
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_out", "proj_in",
        "ff.net.2", "ff.net.0.proj",
    ]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        if "adapter" in n or "wave" in n or "pix_adapter" in n or "sem_adapter" in n:
            continue
        for pat in attn_patterns:
            if pat in n:
                module_name = n.replace(".weight", "")
                if module_name not in l_attn:
                    l_attn.append(module_name)
                break

    if l_attn:
        lora_conf = LoraConfig(
            r=lora_rank,
            init_lora_weights="gaussian",
            target_modules=l_attn,
        )
        unet.add_adapter(lora_conf, adapter_name="attn_lora")

    return l_attn


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

# ★ 改：import v2版本
from osediff_vae_unet_loss_hybrid_wavelora_hl import (
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
                    wave_dim=None, wave_res=32, mlp_ratio=1.0, wave_scale=0.2,
                    return_module_names=False):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    pix_wave_modules, sem_wave_modules = inject_dual_wave_to_unet(
        unet, wave_dim=wave_dim, res=wave_res,
        mlp_ratio=mlp_ratio, scale=wave_scale)

    l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []

    attn_patterns = ["to_k", "to_q", "to_v", "to_out.0",
                     "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]

    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
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



class PiSASR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        wave_dim   = getattr(args, 'wave_dim',   None)
        wave_res   = getattr(args, 'wave_res',   32)
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
                self.unet, wave_dim=wave_dim, res=wave_res,
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
        self._freeze_sem_wave()



    def _freeze_sem_wave(self):
        for blk in self.sem_wave_modules.values():
            blk.requires_grad_(False)
            blk.scale.data.fill_(0.0)

    def _unfreeze_sem_wave(self):
        for blk in self.sem_wave_modules.values():
            blk.train()
            blk.requires_grad_(True)
            blk.scale.data.fill_(self._wave_scale_train)

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


    def set_train_pix(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "pix" in n: _p.requires_grad = True
            if "sem" in n: _p.requires_grad = False
        self._unfreeze_pix_wave()
        self._freeze_sem_wave()

        # ★ 新增：Stage1冻结所有freq_gain
        for blk in self.pix_wave_modules.values():
            blk.freeze_freq_gain()
        for blk in self.sem_wave_modules.values():
            blk.freeze_freq_gain()

    def set_train_sem(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n: _p.requires_grad = True
            if "pix" in n: _p.requires_grad = False
        self._freeze_pix_wave()
        self._unfreeze_sem_wave()

        for blk in self.sem_wave_modules.values():
            blk.unfreeze_freq_gain()
        for blk in self.pix_wave_modules.values():
            blk.freeze_freq_gain()

    # ── checkpoint（完全不动）────────────────────────────────────────────

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

        self._load_wave_after_merge()
        self._move_models_to_device_and_dtype()

        self.ori_unet_weight = {}
        for name, param in self.unet.named_parameters():
            self.ori_unet_weight[name] = param.clone().data.to(
                self.weight_dtype).to("cuda")

        self.timesteps1 = torch.tensor([1], device=self.device).long()
        self.lambda_pix = torch.tensor([args.lambda_pix], device=self.device)
        self.lambda_sem = torch.tensor([args.lambda_sem], device=self.device)

    def _load_wave_after_merge(self):
        if not hasattr(self, '_pending_wave_sd') or self._pending_wave_sd is None:
            self.pix_wave_modules = {}
            self.sem_wave_modules = {}
            return
        cfg = self._pending_wave_cfg
        self.pix_wave_modules, self.sem_wave_modules = inject_dual_wave_to_unet(
            self.unet,
            wave_dim=cfg.get("wave_dim"),
            res=cfg.get("wave_res", 32),
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
            blk.scale.data.fill_(scale)

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
        model_pred_full = self.unet(encoded_control, self.timesteps1,
                                   encoder_hidden_states=prompt_embeds).sample
        self._apply_ori_weight()
        self._apply_sem_wave_scale(0.0)
        model_pred_pix = self.unet(encoded_control, self.timesteps1,
                                   encoder_hidden_states=prompt_embeds).sample
        self._apply_lora_delta()
        self._apply_sem_wave_scale(self.args.wave_scale
                                   if hasattr(self.args, 'wave_scale') else 0.2)
        delta = model_pred_full - model_pred_pix
        return self.lambda_pix * model_pred_pix + self.lambda_sem * delta

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
                        model_out_full = self.unet(t, self.timesteps1,
                                                  encoder_hidden_states=prompt_embeds).sample
                        self._apply_ori_weight()
                        self._apply_sem_wave_scale(0.0)
                        model_out_pix = self.unet(t, self.timesteps1,
                                                  encoder_hidden_states=prompt_embeds).sample
                        self._apply_lora_delta()
                        self._apply_sem_wave_scale(
                            self.args.wave_scale if hasattr(self.args, 'wave_scale') else 0.2)
                        delta = model_out_full - model_out_pix
                        model_out = (self.lambda_pix * model_out_pix
                                     + self.lambda_sem * delta)
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
"""
train_pisasr_WaveLoRA.py

基于 train_pisasr.py，import 换成 pisasr_WaveLoRA，其他逻辑完全一致。
optimizer 需要同时包含：
  - attention LoRA 参数（pix 或 sem）
  - WaveAdapter 参数（pix 或 sem）

修改记录 (2026-04-05):
  - Stage2 切换时 lr 降为原来的 1/2
"""
import os
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "3600000"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
import gc
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from pisasr_wavelora_hl import CSDLoss, PiSASR
from src.my_utils.training_utils import parse_args
from src.datasets.dataset import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import random


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_pisasr = PiSASR(args)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pisasr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available")

    if args.gradient_checkpointing:
        net_pisasr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # Stage1 初始设置
    net_pisasr.unet.set_adapter(
        ['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'])
    net_pisasr.set_train_pix()

    # optimizer：pix attention LoRA + pix WaveAdapter
    layers_to_opt = []
    for n, _p in net_pisasr.unet.named_parameters():
        if "lora" in n and "pix" in n:
            layers_to_opt.append(_p)
    layers_to_opt.extend(net_pisasr.get_pix_wave_params())

    optimizer = torch.optim.AdamW(
        layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val   = PairedSROnlineTxtDataset(split="test",  args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0)

    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained='src/ram_pretrain_model/ram_swin_large_14m.pth',
              pretrained_condition=None, image_size=384, vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    net_pisasr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_pisasr, optimizer, dl_train, lr_scheduler)
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":   weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    global_step  = 0
    lambda_l2    = args.lambda_l2
    lambda_lpips = 0
    lambda_csd   = 0

    if args.resume_ckpt is not None:
        args.pix_steps = 1

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_pisasr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                x_tgt_ram = ram_transforms(x_tgt * 0.5 + 0.5)
                with torch.no_grad():
                    caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{c}, {args.pos_prompt_csd}' for c in caption]

                if global_step == args.pix_steps:
                    if args.is_module:
                        net_pisasr.module.unet.set_adapter([
                            'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
                            'default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
                        net_pisasr.module.set_train_sem()
                    else:
                        net_pisasr.unet.set_adapter([
                            'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
                            'default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
                        net_pisasr.set_train_sem()

                    # Stage2 切换 optimizer 参数：sem attention LoRA + sem WaveAdapter
                    layers_to_opt.clear()
                    for n, _p in accelerator.unwrap_model(net_pisasr).unet.named_parameters():
                        if "lora" in n and "sem" in n:
                            layers_to_opt.append(_p)
                    layers_to_opt.extend(
                        accelerator.unwrap_model(net_pisasr).get_sem_wave_params())
                    optimizer.param_groups[0]['params'] = layers_to_opt

                    # ★ Stage2 lr 降为原来的 1/2
                    stage2_lr = args.learning_rate * 1
                
                    for pg in optimizer.param_groups:
                        pg['lr'] = stage2_lr
                    print(f"[Stage2] lr降为 {stage2_lr} (原始lr的1/5)")

                    lambda_l2    = args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_csd   = args.lambda_csd

                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = \
                    net_pisasr(x_src, x_tgt, batch=batch, args=args)

                loss_l2    = F.mse_loss(x_tgt_pred.float(), x_tgt.float(),
                                        reduction="mean") * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(),
                                       x_tgt.float()).mean() * lambda_lpips
                loss_csd   = net_csd.cal_csd(
                    latents_pred, prompt_embeds, neg_prompt_embeds, args) * lambda_csd
                loss = loss_l2 + loss_lpips + loss_csd

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_csd":   loss_csd.detach().item(),
                        "loss_l2":    loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints",
                                            f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pisasr).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(args.output_dir, "eval",
                                                  f"fid_{global_step}"), exist_ok=True)
                        for step_val, batch_val in enumerate(dl_val):
                            x_src_v    = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt_v    = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            assert x_src_v.shape[0] == 1
                            with torch.no_grad():
                                x_src_ram = ram_transforms(x_src_v * 0.5 + 0.5)
                                caption_v = inference(
                                    x_src_ram.to(dtype=torch.float16), RAM)
                                batch_val["prompt"] = caption_v
                                x_tgt_pred_v, _, _, _ = accelerator.unwrap_model(
                                    net_pisasr)(x_src_v, x_tgt_v,
                                               batch=batch_val, args=args)
                                output_pil  = transforms.ToPILImage()(
                                    x_tgt_pred_v[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(
                                    x_src_v[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(
                                        target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(
                                        target=output_pil, source=input_image)
                                outf = os.path.join(args.output_dir, "eval",
                                                    f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)

                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
'''
tmux new -s pisasr_wavelora_hl1_lr1 -d "CUDA_VISIBLE_DEVICES=\"0\" accelerate launch \
--main_process_port 22809 \
--num_processes 1 \
train_pisasr_wavelora_hl.py \
--pretrained_model_path=/data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base \
--pretrained_model_path_csd=/data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base \
--dataset_txt_paths=/data/datasets/LSDIR/actual_image_paths.txt \
--highquality_dataset_txt_paths=/data/datasets/LSDIR/musiq76_paths.txt \
--dataset_test_folder=preset/testfolder \
--learning_rate=5e-5 \
--train_batch_size=2 \
--prob=0.1 \
--gradient_accumulation_steps=4 \
--enable_xformers_memory_efficient_attention \
--checkpointing_steps=500 \
--seed=123 \
--output_dir=experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-lr1 \
--cfg_csd=7.5 \
--timesteps1=1 \
--lambda_lpips=2.0 \
--lambda_l2=1.0 \
--lambda_csd=1.0 \
--pix_steps=4000 \
--lora_rank_unet_pix=4 \
--lora_rank_unet_sem=4 \
--min_dm_step_ratio=0.02 \
--max_dm_step_ratio=0.5 \
--null_text_ratio=0.5 \
--align_method=adain \
--deg_file_path=params.yml \
--tracker_project_name=PiSASR-wavelora-hl \
--mixed_precision=fp16 \
--max_train_steps=20000 \
--is_module=False \
2>&1 | tee pisasr_wavelora_hl1_lr1.log"
'''
