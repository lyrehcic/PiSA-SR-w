
"""
这个名称叫osediff_wave_hl
含线性AB+hl
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from peft import LoraConfig


# ── ABLinear：低秩 Linear，in → rank → out ─────────────────────────────────

class ABLinear(nn.Module):
    """用 A(in→rank) + B(rank→out) 替代 Linear(in→out)。
    默认 B 零初始化，保证模块刚注入时输出为 0（残差安全）。
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, bias: bool = True):
        super().__init__()
        self.A = nn.Linear(in_features, rank,        bias=False)
        self.B = nn.Linear(rank,        out_features, bias=bias)
        # A：kaiming 初始化；B：零初始化 → 初始整体输出为 0
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        if bias:
            nn.init.zeros_(self.B.bias)

    def forward(self, x):
        return self.B(self.A(x))


# ── ABConv1x1：低秩 Conv2d 1x1，in_ch → rank → out_ch ────────────────────

class ABConv1x1(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 rank: int = 16, bias_on_B: bool = True, zero_init: bool = False):
        super().__init__()
        self.A = nn.Conv2d(in_channels, rank,         1, bias=False)
        self.B = nn.Conv2d(rank,        out_channels, 1, bias=bias_on_B)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        if zero_init:
            nn.init.zeros_(self.B.weight)
            if bias_on_B:
                nn.init.zeros_(self.B.bias)
        else:
            nn.init.zeros_(self.B.weight)  
            if bias_on_B:
                nn.init.zeros_(self.B.bias)

    def forward(self, x):
        return self.B(self.A(x))


# ── 频率增益初始化 ─────────────────────────────────────────────────────────

def _make_freq_gain_init(res: int) -> torch.Tensor:
    import numpy as np
    fh = np.arange(res, dtype=np.float32).reshape(-1, 1)
    fw = np.arange(res, dtype=np.float32).reshape(1, -1)
    # 到 (0,0) 的距离
    freq_dist = np.sqrt(fh**2 + fw**2) / (res * np.sqrt(2))
    # 初始增益：低频=1.0，高频=1.1
    gain = 1.0 + 0.1 * freq_dist
    return torch.tensor(gain, dtype=torch.float32).unsqueeze(-1)  # [res, res, 1]


# ── Wave2D_Fixed ───────────────────────────────────────────────────────────

class Wave2D_Fixed(nn.Module):
    """
    ABLinear(rank=16)：
        linear     : dim → 2*dim
        gate_proj  : dim → dim
        out_linear : dim → dim
        to_k[0]    : dim → dim
    """
    def __init__(self, dim: int, res: int = 64, inner_rank: int = 16):
        super().__init__()
        self.dim = dim
        self.res = res

        self.linear     = ABLinear(dim, 2 * dim, rank=inner_rank, bias=True)
        self.gate_proj  = ABLinear(dim, dim,     rank=inner_rank, bias=True)
        self.out_norm   = nn.LayerNorm(dim)
        self.out_linear = ABLinear(dim, dim,     rank=inner_rank, bias=True)
        self.to_k = nn.Sequential(
            ABLinear(dim, dim, rank=inner_rank, bias=True),
            nn.GELU(),
        )

        self.c     = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

        # Stage1 冻结，Stage2 sem 解冻
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
        fg = self.freq_gain
        if (H, W) == (fg.shape[0], fg.shape[1]):
            return fg
        fg_4d = fg.permute(2, 0, 1).unsqueeze(0).float()   # [1,1,res,res]
        fg_interp = F.interpolate(
            fg_4d, size=(H, W), mode='bilinear', align_corners=False
        )                                                    # [1,1,H,W]
        return fg_interp.squeeze(0).permute(1, 2, 0).to(fg.dtype)  # [H,W,1]

    def forward(self, x: torch.Tensor, freq_embed=None):
        orig_dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1).contiguous()          # [B,H,W,C]

        xz = self.linear.float()(x_cl)                     # ABLinear: dim→16→2*dim
        x_disp, z_vel = xz.chunk(2, dim=-1)
        v0 = F.silu(z_vel)

        cosH, cosW = self._get_cos_maps(H, W, x.device)
        u0_hat = self._dct2d(x_disp, cosH, cosW)
        v0_hat = self._dct2d(v0,     cosH, cosW)
        u0_hat = torch.clamp(u0_hat, -100.0, 100.0)
        v0_hat = torch.clamp(v0_hat, -100.0, 100.0)

        if freq_embed is not None:
            fe = freq_embed.unsqueeze(0).expand(B, -1, -1, -1).float()
            t  = self.to_k.float()(fe)                     # ABLinear + GELU
        else:
            t = torch.zeros(B, H, W, self.dim, device=x.device, dtype=torch.float32)

        c_safe  = torch.abs(self.c.float()) + 1e-4
        alpha_s = torch.clamp(self.alpha.float(), min=0.0)
        ct      = torch.clamp(c_safe * t, -20.0, 20.0)

        u_hat = (torch.cos(ct) * u0_hat
                 + torch.sin(ct) / c_safe * (v0_hat + alpha_s / 2.0 * u0_hat))

        # freq_gain: [H,W,1] 广播到 [B,H,W,C]
        gain  = self._get_freq_gain(H, W).float()
        u_hat = u_hat * gain

        x_out = self._idct2d(u_hat, cosH, cosW)
        x_out = torch.clamp(x_out, -100.0, 100.0)
        x_out = self.out_norm.float()(x_out)

        gate  = F.silu(self.gate_proj.float()(x_disp))    # ABLinear: dim→16→dim
        x_out = x_out * gate
        x_out = self.out_linear.float()(x_out)             # ABLinear: dim→16→dim
        x_out = torch.nan_to_num(x_out, nan=0.0)

        return x_out.permute(0, 3, 1, 2).contiguous().to(orig_dtype)


# ── WaveAdapter ────────────────────────────────────────────────────────────

class WaveAdapter(nn.Module):
    """
    Wave 适配器。
    - proj_in  : Conv2d(channels, wave_dim, 1)      → ABConv1x1(rank=16)
    - proj_out : Conv2d(wave_dim, channels, 1)      → ABConv1x1(rank=16, 零初始化)
    - ffn      : 原两个大Conv 改为 AB低秩分解(rank=16)：
                   A1(channels→16)→B1(16→hidden)→GELU→A2(hidden→16)→B2(16→channels)
                 B2 零初始化
    """
    def __init__(self, channels: int, wave_dim: int = None,
                 res: int = 64, mlp_ratio: float = 1.0, scale: float = 0.2,
                 rank: int = 16):
        super().__init__()
        self.channels = channels
        self.scale    = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)
        wave_dim      = wave_dim or channels

        self.norm_in = nn.GroupNorm(min(32, channels), channels, eps=1e-6)

        # proj_in: channels → wave_dim（ABConv1x1，rank=16）
        if wave_dim != channels:
            self.proj_in = ABConv1x1(channels, wave_dim, rank=rank,
                                     bias_on_B=False, zero_init=False)
        else:
            self.proj_in = nn.Identity()

        # Wave 模块（内部 Linear 已是 ABLinear）
        self.wave = Wave2D_Fixed(dim=wave_dim, res=res, inner_rank=rank)


        self.proj_out = ABConv1x1(wave_dim, channels, rank=rank,
                                  bias_on_B=True, zero_init=True)

        #Conv2d(channels,hidden,1) → GELU → Conv2d(hidden,channels,1)
        #A1(channels→rank)→B1(rank→hidden)→GELU→A2(hidden→rank)→B2(rank→channels)

        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, rank,   1, bias=False),   # A1
            nn.Conv2d(rank,   hidden,   1, bias=True),    # B1
            nn.GELU(),
            nn.Conv2d(hidden,   rank,   1, bias=False),   # A2
            nn.Conv2d(rank,   channels, 1, bias=True),    # B2 
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

        self.freq_embed = nn.Parameter(torch.zeros(res, res, wave_dim))
        trunc_normal_(self.freq_embed, std=0.02)

    def freeze_freq_gain(self):
        self.wave.freq_gain.requires_grad = False

    def unfreeze_freq_gain(self):
        self.wave.freq_gain.requires_grad = True

    def _get_freq(self, H, W):
        fe = self.freq_embed
        if (H, W) == (fe.shape[0], fe.shape[1]):
            return fe
        fe_4d     = fe.permute(2, 0, 1).unsqueeze(0).float()
        fe_interp = F.interpolate(fe_4d, size=(H, W), mode='bilinear', align_corners=False)
        return fe_interp.squeeze(0).permute(1, 2, 0).to(fe.dtype)

    def forward(self, x: torch.Tensor):
        _, _, H, W = x.shape
        x_norm = self.norm_in(x)

        r = self.proj_in(x_norm)                 # ABConv1x1
        r = self.wave(r, self._get_freq(H, W))
        r = self.proj_out(r)                     # ABConv1x1

        f = self.ffn(x_norm)                     # AB低秩FFN
        return (r + f) * self.scale


# ── _DualWaveConv ──────────────────────────────────────────────────────────

class _DualWaveConv(nn.Module):
    def __init__(self, orig_conv, pix_adapter, sem_adapter):
        super().__init__()
        self.conv        = orig_conv
        self.pix_adapter = pix_adapter
        self.sem_adapter = sem_adapter

    def forward(self, x, scale=None):
        h = self.conv(x)
        return h + self.pix_adapter(h) + self.sem_adapter(h)


# ── inject_dual_wave_to_unet ───────────────────────────────────────────────
# 签名与原版完全一致，文件6无需修改

def inject_dual_wave_to_unet(unet, wave_dim=None, res=64,
                              mlp_ratio=1.0, scale=0.2,
                              rank: int = 16) -> tuple:
    pix_modules = {}
    sem_modules = {}

    def _blk(out_ch):
        wd = wave_dim or min(out_ch, 512)
        pix = WaveAdapter(channels=out_ch, wave_dim=wd,
                          res=res, mlp_ratio=mlp_ratio, scale=scale,
                          rank=rank)
        sem = WaveAdapter(channels=out_ch, wave_dim=wd,
                          res=res, mlp_ratio=mlp_ratio, scale=0.0,
                          rank=rank)
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


# ── add_lora_to_unet_attention ─────────────────────────────────────────────

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