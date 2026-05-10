"""
osediff_wave_hl_DE.py

在原版 osediff_wave_hl.py 基础上改进 WaveAdapter：
  1. freq_embed 分辨率从 res=64 降到 res=16（参数量从 2M/个 → 0.13M/个）
  2. 新增 D×E 自适应调制：
       - D：从输入特征图提取 Sobel + Laplacian + LocalVariance 细节图
       - E：局部高频能量比（avg_pool 差值法）
       - DE = D ⊙ E，blur 后归一化到 [0,1]
  3. adaptive_freq = base_freq × (1 + hf_scale·DE − lf_scale·(1−DE))
       - DE 高（边缘/纹理）→ 增强 Wave 演化（高频恢复）
       - DE 低（平坦/低频）→ 抑制 Wave 演化（结构保护）
  4. hf_scale / lf_scale 为可学习标量，初始值 0.5 / 0.1

其余所有模块（ABLinear, ABConv1x1, Wave2D_Fixed, _DualWaveConv,
inject_dual_wave_to_unet, add_lora_to_unet_attention）完全不变。

总参数量：约 1.32B（原版 1.57B），接近原版 PiSASR 的 1.30B。

线性变AB,含hl+fiesr的DE
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
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        if bias:
            nn.init.zeros_(self.B.bias)

    def forward(self, x):
        return self.B(self.A(x))


# ── ABConv1x1：低秩 Conv2d 1x1，in_ch → rank → out_ch ────────────────────

class ABConv1x1(nn.Module):
    """用两个 1x1 Conv 替代一个 Conv2d(in, out, 1)。
    zero_init=True 时 B 零初始化，用于 proj_out 保证初始残差为 0。
    """
    def __init__(self, in_channels: int, out_channels: int,
                 rank: int = 16, bias_on_B: bool = True, zero_init: bool = False):
        super().__init__()
        self.A = nn.Conv2d(in_channels, rank,         1, bias=False)
        self.B = nn.Conv2d(rank,        out_channels, 1, bias=bias_on_B)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
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
    freq_dist = np.sqrt(fh**2 + fw**2) / (res * np.sqrt(2))
    gain = 1.0 + 0.1 * freq_dist
    return torch.tensor(gain, dtype=torch.float32).unsqueeze(-1)  # [res, res, 1]


# ── Wave2D_Fixed ───────────────────────────────────────────────────────────

class Wave2D_Fixed(nn.Module):
    """
    波动方程 2D 模块（不变）。
    内部所有 Linear 改为 ABLinear(rank=16)。
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
        fg_4d = fg.permute(2, 0, 1).unsqueeze(0).float()
        fg_interp = F.interpolate(
            fg_4d, size=(H, W), mode='bilinear', align_corners=False
        )
        return fg_interp.squeeze(0).permute(1, 2, 0).to(fg.dtype)

    def forward(self, x: torch.Tensor, freq_embed=None):
        """
        freq_embed: [B, H, W, dim] 或 [H, W, dim]
        改进版传入的是 adaptive_freq [B, H, W, dim]，直接支持 batch 维度。
        """
        orig_dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1).contiguous()          # [B,H,W,C]

        xz = self.linear.float()(x_cl)
        x_disp, z_vel = xz.chunk(2, dim=-1)
        v0 = F.silu(z_vel)

        cosH, cosW = self._get_cos_maps(H, W, x.device)
        u0_hat = self._dct2d(x_disp, cosH, cosW)
        v0_hat = self._dct2d(v0,     cosH, cosW)
        u0_hat = torch.clamp(u0_hat, -100.0, 100.0)
        v0_hat = torch.clamp(v0_hat, -100.0, 100.0)

        if freq_embed is not None:
            fe = freq_embed.float()
            # 支持 [H,W,dim]（原版）和 [B,H,W,dim]（改进版）两种输入
            if fe.dim() == 3:
                fe = fe.unsqueeze(0).expand(B, -1, -1, -1)
            t = self.to_k.float()(fe)                       # [B,H,W,dim]
        else:
            t = torch.zeros(B, H, W, self.dim,
                            device=x.device, dtype=torch.float32)

        c_safe  = torch.abs(self.c.float()) + 1e-4
        alpha_s = torch.clamp(self.alpha.float(), min=0.0)
        ct      = torch.clamp(c_safe * t, -20.0, 20.0)

        u_hat = (torch.cos(ct) * u0_hat
                 + torch.sin(ct) / c_safe * (v0_hat + alpha_s / 2.0 * u0_hat))

        gain  = self._get_freq_gain(H, W).float()
        u_hat = u_hat * gain

        x_out = self._idct2d(u_hat, cosH, cosW)
        x_out = torch.clamp(x_out, -100.0, 100.0)
        x_out = self.out_norm.float()(x_out)

        gate  = F.silu(self.gate_proj.float()(x_disp))
        x_out = x_out * gate
        x_out = self.out_linear.float()(x_out)
        x_out = torch.nan_to_num(x_out, nan=0.0)

        return x_out.permute(0, 3, 1, 2).contiguous().to(orig_dtype)


# ── WaveAdapter（改进版，含 D×E 自适应 freq_embed）────────────────────────

class WaveAdapter(nn.Module):
    """
    改进版 WaveAdapter。

    相比原版的两处关键改动：

    [1] freq_embed 分辨率降低：
        res=16（原 64），参数从 64×64×wave_dim 降到 16×16×wave_dim。
        大多数 U-Net 特征图本来就在 8×8~32×32 之间，res=64 严重浪费。
        损失的空间精度由 D×E 实时细节图补偿。

    [2] D×E 自适应调制（_get_DE_map + adaptive_freq）：
        D = Sobel + Laplacian + LocalVariance（细节丰富程度）
        E = 局部高频能量比（avg_pool 差值）
        DE = D ⊙ E，blur，归一化到 [0,1]

        adaptive_freq = base_freq × (1 + hf_scale·DE − lf_scale·(1−DE))
          - DE 高（边缘/纹理区域）→ Wave 演化增强 → 高频细节恢复
          - DE 低（平坦/低频区域）→ Wave 演化抑制 → 结构保护，避免过处理

        hf_scale / lf_scale 为可学习标量，初始值 0.5 / 0.1。
        D×E 计算在 torch.no_grad() 下执行，不增加反向传播开销。
    """
    def __init__(self, channels: int, wave_dim: int = None,
                 res: int = 16,           # ← 从 64 改为 16
                 mlp_ratio: float = 1.0, scale: float = 0.2,
                 rank: int = 16):
        super().__init__()
        self.channels = channels
        self.scale    = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)
        wave_dim      = wave_dim or channels

        self.norm_in = nn.GroupNorm(min(32, channels), channels, eps=1e-6)

        if wave_dim != channels:
            self.proj_in = ABConv1x1(channels, wave_dim, rank=rank,
                                     bias_on_B=False, zero_init=False)
        else:
            self.proj_in = nn.Identity()

        self.wave = Wave2D_Fixed(dim=wave_dim, res=res, inner_rank=rank)

        self.proj_out = ABConv1x1(wave_dim, channels, rank=rank,
                                  bias_on_B=True, zero_init=True)

        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, rank,   1, bias=False),
            nn.Conv2d(rank,   hidden,   1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden,   rank,   1, bias=False),
            nn.Conv2d(rank,   channels, 1, bias=True),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

        # freq_embed：低分辨率可学习基础频率嵌入
        # res=16 → 16×16×wave_dim = 0.13M（原 64×64×wave_dim = 2M）
        self.freq_embed = nn.Parameter(torch.zeros(res, res, wave_dim))
        trunc_normal_(self.freq_embed, std=0.02)

        # D×E 调制强度（可学习标量）
        # hf_scale > 0：高频区域增强 Wave 演化
        # lf_scale > 0：低频区域抑制 Wave 演化
        self.hf_scale = nn.Parameter(torch.tensor(0.5))
        self.lf_scale = nn.Parameter(torch.tensor(0.1))

        # 用于 D 计算的固定卷积核（注册为 buffer，不参与训练）
        kx = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2).contiguous()
        lap = torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
        ).view(1, 1, 3, 3)
        k3  = torch.ones(1, 1, 3, 3) / 9.0

        self.register_buffer('_kx',  kx,  persistent=False)
        self.register_buffer('_ky',  ky,  persistent=False)
        self.register_buffer('_lap', lap, persistent=False)
        self.register_buffer('_k3',  k3,  persistent=False)

    # ------------------------------------------------------------------
    # freq_embed 工具
    # ------------------------------------------------------------------

    def freeze_freq_gain(self):
        self.wave.freq_gain.requires_grad = False

    def unfreeze_freq_gain(self):
        self.wave.freq_gain.requires_grad = True

    def _get_base_freq(self, H: int, W: int) -> torch.Tensor:
        """把低分辨率 freq_embed 插值到当前特征图尺寸 [H,W,wave_dim]。"""
        fe = self.freq_embed                                  # [res,res,dim]
        if (H, W) == (fe.shape[0], fe.shape[1]):
            return fe
        fe_4d     = fe.permute(2, 0, 1).unsqueeze(0).float() # [1,dim,res,res]
        fe_interp = F.interpolate(
            fe_4d, size=(H, W), mode='bilinear', align_corners=False
        )                                                     # [1,dim,H,W]
        return fe_interp.squeeze(0).permute(1, 2, 0).to(fe.dtype)  # [H,W,dim]

    # ------------------------------------------------------------------
    # D×E 自适应细节图
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_DE_map(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        从归一化后的输入特征图计算 DE [B,1,H,W]，值域 [0,1]。

        D（细节丰富度）：
            Sobel + Laplacian + LocalVariance 的均值，
            对应 FiDeSR Algorithm 1 中的 Detail Map D。

        E（高频能量比）：
            局部高频能量 / 局部总能量，
            替代 FiDeSR 中需要 GT 的 Error Map E。
            训练和推理时都可用，不依赖 GT。

        DE = D ⊙ E，经过 3×3 blur 平滑后归一化。

        注意：全程强制 float32，因为 quantile() 不支持 fp16/bf16。
        最后转回 x_norm.dtype 返回。
        """
        orig_dtype = x_norm.dtype

        # ★ 关键：所有中间计算强制 float32
        # x_norm 在 fp16 混合精度下是 fp16，mean() 结果也是 fp16
        # 必须在 .float() 之后再做 conv2d，且 kernel 也要 float32
        x_gray = x_norm.mean(dim=1, keepdim=True).float()    # [B,1,H,W] float32
        B      = x_gray.shape[0]

        # kernel 强制 float32（buffer 本来是 float32，但保险起见显式转）
        kx  = self._kx.float()
        ky  = self._ky.float()
        lap = self._lap.float()
        k3  = self._k3.float()

        # ── D：细节图 ──────────────────────────────────────────────────
        gx    = F.conv2d(x_gray, kx,  padding=1)
        gy    = F.conv2d(x_gray, ky,  padding=1)
        sobel = (gx ** 2 + gy ** 2 + 1e-8).sqrt()            # [B,1,H,W]

        laplacian = F.conv2d(x_gray, lap, padding=1).abs()

        mu    = F.avg_pool2d(x_gray,    7, stride=1, padding=3)
        mu2   = F.avg_pool2d(x_gray**2, 7, stride=1, padding=3)
        var   = (mu2 - mu**2).clamp(min=0)

        D = (sobel + laplacian + var) / 3.0
        # quantile 归一化（此时 D 已是 float32，quantile 可正常执行）
        D_flat = D.reshape(B, -1)
        q99    = D_flat.quantile(0.99, dim=1).clamp(min=1e-6).view(B, 1, 1, 1)
        D      = (D / q99).clamp(0, 1)
        D      = F.conv2d(D, k3, padding=1)                   # 3×3 blur

        # ── E：高频能量比 ──────────────────────────────────────────────
        lf    = F.avg_pool2d(x_gray, 7, stride=1, padding=3)
        hf    = (x_gray - lf).abs()
        total = x_gray.abs() + 1e-6
        E     = hf / total
        E_flat = E.reshape(B, -1).float()
        q99_e  = E_flat.quantile(0.99, dim=1).clamp(min=1e-6).view(B, 1, 1, 1)
        E      = (E / q99_e).clamp(0, 1)

        # ── DE = D ⊙ E，blur ──────────────────────────────────────────
        DE      = D * E
        DE      = F.conv2d(DE, k3, padding=1)
        DE_flat = DE.reshape(B, -1).float()
        q99_de  = DE_flat.quantile(0.99, dim=1).clamp(min=1e-6).view(B, 1, 1, 1)
        DE      = (DE / q99_de).clamp(0, 1)

        # 最后转回原始 dtype（fp16/bf16），与后续 adaptive_freq 计算兼容
        return DE.to(orig_dtype)                              # [B,1,H,W]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        x_norm = self.norm_in(x)                              # [B,C,H,W]

        # ── 1. 计算 D×E 细节图（no_grad）─────────────────────────────
        DE = self._get_DE_map(x_norm)                         # [B,1,H,W]
        DE_bhwc = DE.permute(0, 2, 3, 1)                     # [B,H,W,1]
        lf_mask = 1.0 - DE_bhwc                               # 低频区域权重

        # ── 2. 构造 adaptive_freq [B,H,W,wave_dim] ───────────────────
        base_freq = self._get_base_freq(H, W)                 # [H,W,wave_dim]
        # hf_scale / lf_scale 通过 clamp 保证非负，避免反向传播时符号翻转
        hf = torch.clamp(self.hf_scale, min=0.0)
        lf = torch.clamp(self.lf_scale, min=0.0)

        adaptive_freq = base_freq.unsqueeze(0) * (
            1.0
            + hf * DE_bhwc      # 高频区域（DE高）：增强波动方程演化
            - lf * lf_mask      # 低频区域（DE低）：抑制波动方程演化
        )                                                     # [B,H,W,wave_dim]

        # ── 3. Wave 路径 ──────────────────────────────────────────────
        r = self.proj_in(x_norm)                              # ABConv1x1 或 Identity
        r = self.wave(r, adaptive_freq)                       # Wave2D_Fixed
        r = self.proj_out(r)                                  # ABConv1x1（零初始化）

        # ── 4. FFN 路径 ───────────────────────────────────────────────
        f = self.ffn(x_norm)

        return (r + f) * self.scale


# ── _DualWaveConv ──────────────────────────────────────────────────────────

class _DualWaveConv(nn.Module):
    """不变。"""
    def __init__(self, orig_conv, pix_adapter, sem_adapter):
        super().__init__()
        self.conv        = orig_conv
        self.pix_adapter = pix_adapter
        self.sem_adapter = sem_adapter

    def forward(self, x, scale=None):
        h = self.conv(x)
        return h + self.pix_adapter(h) + self.sem_adapter(h)


# ── inject_dual_wave_to_unet ───────────────────────────────────────────────

def inject_dual_wave_to_unet(unet, wave_dim=None, res=16,   # ← 默认 res=16
                              mlp_ratio=1.0, scale=0.2,
                              rank: int = 16) -> tuple:
    """
    签名与原版完全一致（res 默认值从 64 改为 16）。
    pisasr_wave_hl.py 无需修改。
    """
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
    """不变。"""
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