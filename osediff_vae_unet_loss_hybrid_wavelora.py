"""
osediff_vae_unet_loss_hybrid_WaveLoRA.py

设计原则（数学严谨版）：
  conv 层：用双 WaveAdapter（pix / sem）替代 LoRA
           输入是 feature map [B,C,H,W]，有空间结构，DCT 全局变换有物理意义
  attention 层：保留 LoRA
                输入是 token 序列 [B,L,C]，是语义空间，DCT 没有物理意义
                LoRA 在语义空间做低秩调整是合理的

两条路径完全正交，互不干扰：
  - WaveAdapter 作用于 conv 输出的 feature map
  - LoRA 作用于 attention 的权重矩阵

双 WaveAdapter 的分工：
  - pix_adapter：Stage1 训练，负责保真（PSNR）
  - sem_adapter：Stage2 训练，负责感知（MUSIQ/CLIPIQA）
  推理时两次前向，差值机制和原版 PiSASR 完全一样

Wave2D_Fixed 使用原版实现（经过验证，数值稳定）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from peft import LoraConfig


# ── Wave2D_Fixed（原版，经过验证）─────────────────────────────────────────

class Wave2D_Fixed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim        = dim
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
        x_out = self._idct2d(u_hat, cosH, cosW)
        x_out = torch.clamp(x_out, -100.0, 100.0)
        x_out = self.out_norm.float()(x_out)
        gate  = F.silu(self.gate_proj.float()(x_disp))
        x_out = x_out * gate
        x_out = self.out_linear.float()(x_out)
        x_out = torch.nan_to_num(x_out, nan=0.0)
        return x_out.permute(0, 3, 1, 2).contiguous().to(orig_dtype)


# ── WaveAdapter（单个，pix 或 sem 各一个）────────────────────────────────

class WaveAdapter(nn.Module):
    """
    作用于 conv 输出的 feature map [B,C,H,W]。
    零初始化输出，训练初期不干扰原始 conv。
    """
    def __init__(self, channels: int, wave_dim: int = None,
                 res: int = 64, mlp_ratio: float = 1.0, scale: float = 0.2):
        super().__init__()
        self.channels = channels
        self.scale    = scale
        wave_dim      = wave_dim or channels

        self.norm_in  = nn.GroupNorm(min(32, channels), channels, eps=1e-6)
        self.proj_in  = (nn.Conv2d(channels, wave_dim, 1, bias=False)
                         if wave_dim != channels else nn.Identity())
        self.wave     = Wave2D_Fixed(dim=wave_dim)
        self.proj_out = nn.Conv2d(wave_dim, channels, 1, bias=True)

        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

        self.freq_embed = nn.Parameter(torch.zeros(res, res, wave_dim))
        trunc_normal_(self.freq_embed, std=0.02)

        # 零初始化：训练初期 adapter 输出为零，不破坏预训练特征
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

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


# ── _DualWaveConv：每个 conv 位置挂 pix + sem 两个 WaveAdapter ────────────

class _DualWaveConv(nn.Module):
    """
    h = conv(x) + pix_adapter(h) * pix_scale + sem_adapter(h) * sem_scale

    Stage1：pix_adapter 训练（scale=0.2），sem_adapter 冻结（scale=0.0）
    Stage2：sem_adapter 训练（scale=0.2），pix_adapter 冻结（scale 保持）
    推理：两次前向，差值机制和原版 PiSASR 完全一致
    """
    def __init__(self, orig_conv, pix_adapter, sem_adapter):
        super().__init__()
        self.conv        = orig_conv
        self.pix_adapter = pix_adapter
        self.sem_adapter = sem_adapter

    def forward(self, x, scale=None):
        h = self.conv(x)
        return h + self.pix_adapter(h) + self.sem_adapter(h)


# ── 注入函数 ───────────────────────────────────────────────────────────────

def inject_dual_wave_to_unet(unet, wave_dim=None, res=64,
                              mlp_ratio=1.0, scale=0.2) -> tuple:
    """
    在 UNet 所有 conv 层注入双 WaveAdapter（pix + sem）。
    返回 (pix_modules, sem_modules) 两个 dict，key 相同。
    必须在 add_adapter（LoRA）之前调用。
    """
    pix_modules = {}
    sem_modules = {}

    def _blk(out_ch):
        wd = wave_dim or min(out_ch, 512)
        pix = WaveAdapter(channels=out_ch, wave_dim=wd,
                          res=res, mlp_ratio=mlp_ratio, scale=scale)
        sem = WaveAdapter(channels=out_ch, wave_dim=wd,
                          res=res, mlp_ratio=mlp_ratio, scale=0.0)  # sem 初始 scale=0
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
    """
    只在 attention 相关层加 LoRA，不碰 conv 层。
    与 WaveAdapter 的 conv 路径完全正交。
    """
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