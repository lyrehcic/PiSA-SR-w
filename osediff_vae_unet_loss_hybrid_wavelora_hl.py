"""  
  修改记录 (2026-04-10):
    - Wave2D_Fixed 加入 freq_gain 参数
    - freq_gain shape: [res, res, 1]，初始化为低频=1，高频略>1
    - forward 里在波动方程之后、IDCT之前乘以 freq_gain
    - WaveAdapter 加入 freeze_freq_gain / unfreeze_freq_gain 方法
    - PiSASR.set_train_pix / set_train_sem 分别控制 freq_gain 的冻结
    
    有hl,无线性换AB
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