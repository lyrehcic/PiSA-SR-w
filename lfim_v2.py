"""
lfim_v2.py

三频段自适应潜在频率注入模块（Multi-band Latent Frequency Injection Module）

相比 FiDeSR 的两频段 LFIM 的改进：
    1. 三频段分解（低频/中频/高频），中频单独处理
       - 低频 (0 ~ cutoff_low)  ：结构/色调，平坦区域注入
       - 中频 (cutoff_low ~ cutoff_mid)：纹理/边缘，核心创新段
       - 高频 (cutoff_mid ~ 1)  ：细节/噪声，保守注入
    2. 中频空间门控：边缘（Sobel）+ 局部熵双路融合
       - Sobel 捕捉边缘区域
       - 局部熵捕捉纹理复杂区域（草地/布料/毛发等 Sobel 弱但纹理丰富的区域）
       - 两者互补，覆盖更全面的细节区域
    3. 高频空间门控：只用 Sobel（更保守，避免噪声放大）
    4. 低频空间门控：用平坦区域（1 - 中频门控），稳定结构

完全无参数，推理时即插即用，不需要重训练。

用法：
    from lfim_v2 import apply_lfim_v2
    z_enhanced = apply_lfim_v2(
        z         = x_denoised,      # [B,C,H,W] latent
        lq_image  = c_t,             # [B,3,H,W] LQ图像（可选但推荐）
        lf_alpha  = 0.1,             # 低频注入强度
        mf_beta   = 0.3,             # 中频注入强度（主要调节项）
        hf_beta   = 0.1,             # 高频注入强度
    )

推荐扫参策略：
    先固定 lf_alpha=0.0, hf_beta=0.1，扫描 mf_beta=0.1~0.5
    找到最佳 mf_beta 后，再微调 lf_alpha 和 hf_beta
"""

import torch
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# 滤波器
# ══════════════════════════════════════════════════════════════════════════════

def _butterworth_lp(H: int, W: int, cutoff: float, order: int = 2,
                    device=None) -> torch.Tensor:
    """
    Butterworth 低通滤波器 mask [H,W]，值域 [0,1]。
    cutoff: 截止频率比例 0~1
    """
    fh = torch.fft.fftfreq(H, device=device).unsqueeze(1).expand(H, W)
    fw = torch.fft.fftfreq(W, device=device).unsqueeze(0).expand(H, W)
    freq = (fh**2 + fw**2).sqrt()
    lp = 1.0 / (1.0 + (freq / (cutoff + 1e-8)) ** (2 * order))
    return lp.float()


def _bandpass_mask(H: int, W: int,
                   cutoff_low: float, cutoff_high: float,
                   order: int = 2, device=None) -> torch.Tensor:
    """
    带通滤波器 mask [H,W]：保留 cutoff_low ~ cutoff_high 之间的频率。
    = 低通(cutoff_high) - 低通(cutoff_low)
    """
    lp_high = _butterworth_lp(H, W, cutoff_high, order, device)
    lp_low  = _butterworth_lp(H, W, cutoff_low,  order, device)
    bp = (lp_high - lp_low).clamp(min=0)
    return bp


def _highpass_mask(H: int, W: int, cutoff: float,
                   order: int = 2, device=None) -> torch.Tensor:
    """高通滤波器 mask [H,W] = 1 - 低通(cutoff)"""
    return (1.0 - _butterworth_lp(H, W, cutoff, order, device)).clamp(min=0)


# ══════════════════════════════════════════════════════════════════════════════
# 空间门控
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _sobel_gate(x_gray: torch.Tensor) -> torch.Tensor:
    """
    Sobel 边缘门控 [B,1,H,W]，值域 [0,1]。
    x_gray: [B,1,H,W] float32 灰度图
    """
    B = x_gray.shape[0]
    kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],
                       device=x_gray.device).view(1,1,3,3)
    ky = kx.transpose(-1,-2).contiguous()
    gx = F.conv2d(x_gray, kx, padding=1)
    gy = F.conv2d(x_gray, ky, padding=1)
    sobel = (gx**2 + gy**2 + 1e-8).sqrt()
    q = sobel.reshape(B,-1).quantile(0.99, dim=1).clamp(1e-6).view(B,1,1,1)
    return (sobel / q).clamp(0, 1)


@torch.no_grad()
def _entropy_gate(x_gray: torch.Tensor,
                  ksize: int = 9, bins: int = 32) -> torch.Tensor:
    """
    局部熵门控 [B,1,H,W]，值域 [0,1]。
    局部熵高 = 该区域纹理复杂（草地/布料/毛发等 Sobel 弱但细节丰富的区域）。

    实现：用局部像素分布的近似熵（基于局部方差的对数近似），
    避免真实熵计算的高计算开销。

    x_gray: [B,1,H,W] float32，值域任意（内部会归一化）
    ksize:  局部窗口大小
    """
    B = x_gray.shape[0]
    pad = ksize // 2

    # 归一化到 [0,1]
    x_min = x_gray.reshape(B,-1).min(1)[0].view(B,1,1,1)
    x_max = x_gray.reshape(B,-1).max(1)[0].view(B,1,1,1)
    x_norm = (x_gray - x_min) / (x_max - x_min + 1e-6)

    # 局部均值和二阶矩
    mu  = F.avg_pool2d(x_norm,    ksize, stride=1, padding=pad)
    mu2 = F.avg_pool2d(x_norm**2, ksize, stride=1, padding=pad)
    var = (mu2 - mu**2).clamp(min=1e-8)

    # 近似熵：H ≈ 0.5 * log(2πe * σ²)（高斯近似）
    # 单调递增于方差，计算简单
    entropy = 0.5 * torch.log(2 * math.pi * math.e * var)
    entropy = entropy.clamp(min=0)

    # 归一化
    q = entropy.reshape(B,-1).quantile(0.99, dim=1).clamp(1e-6).view(B,1,1,1)
    return (entropy / q).clamp(0, 1)


@torch.no_grad()
def _build_spatial_gates(source: torch.Tensor,
                         entropy_weight: float = 0.5) -> dict:
    """
    从输入图像或 latent 构建三个空间门控。

    source: [B,C,H,W]，可以是 LQ 图像或 latent
    entropy_weight: 局部熵在中频门控中的权重（0~1），
                    其余权重分配给 Sobel
                    默认 0.5，两者各占一半

    返回：
        gates['mf']  [B,1,H,W]  中频门控：Sobel × (1-w) + 熵 × w
        gates['hf']  [B,1,H,W]  高频门控：只用 Sobel（更保守）
        gates['lf']  [B,1,H,W]  低频门控：1 - 中频门控（平坦区域）
    """
    x_gray = source.mean(dim=1, keepdim=True).float()  # [B,1,H,W]

    M_edge    = _sobel_gate(x_gray)                    # 边缘图
    M_entropy = _entropy_gate(x_gray)                  # 局部熵图

    # 中频：边缘 + 局部熵融合（覆盖边缘区域和纹理区域）
    w = entropy_weight
    M_mf = ((1.0 - w) * M_edge + w * M_entropy).clamp(0, 1)

    # 高频：只用边缘（保守，避免纹理区域引入高频噪声）
    M_hf = M_edge

    # 低频：平坦区域（中频门控的补集）
    M_lf = (1.0 - M_mf).clamp(0, 1)

    return {'mf': M_mf, 'hf': M_hf, 'lf': M_lf}


# ══════════════════════════════════════════════════════════════════════════════
# 通道门控
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _channel_gates(z: torch.Tensor,
                   cutoff_low: float,
                   cutoff_mid: float) -> dict:
    """
    基于通道激活方差的三频段通道门控。

    通道激活方差大 → 该通道在空间上变化剧烈 → 携带更多细节/纹理信息。
    比 FiDeSR 的频率能量比更直接反映语义。

    返回：
        gates['mf'] [B,C,1,1]  中频通道门控
        gates['hf'] [B,C,1,1]  高频通道门控
        gates['lf'] [B,C,1,1]  低频通道门控
    """
    B, C, H, W = z.shape
    z_f32 = z.float()

    # 通道空间方差 [B,C]
    ch_var = z_f32.var(dim=[-2,-1])                    # [B,C]

    def _norm_ch(x):
        """归一化到 [0,1]，per-sample"""
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return ((x - x_min) / (x_max - x_min + 1e-8)).view(B, C, 1, 1)

    # 用 FFT 辅助区分低/中/高频通道
    Z = torch.fft.fft2(z_f32)                          # [B,C,H,W] 复数

    lp_low  = _butterworth_lp(H, W, cutoff_low,  device=z.device).view(1,1,H,W)
    lp_high = _butterworth_lp(H, W, cutoff_mid,  device=z.device).view(1,1,H,W)
    bp_mask = (lp_high - lp_low).clamp(0)             # 带通
    hp_mask = (1.0 - lp_high).clamp(0)                # 高通

    energy_total = (Z.abs()**2).mean(dim=[-2,-1]).clamp(1e-8)   # [B,C]
    energy_mf    = (Z.abs()**2 * bp_mask).mean(dim=[-2,-1])      # [B,C]
    energy_hf    = (Z.abs()**2 * hp_mask).mean(dim=[-2,-1])      # [B,C]
    energy_lf    = energy_total - energy_mf - energy_hf

    # 中频通道门控：中频能量占比 × 通道方差（双重加权）
    ratio_mf = energy_mf / energy_total
    ratio_hf = energy_hf / energy_total
    ratio_lf = energy_lf.clamp(0) / energy_total

    # 和方差结合：方差大的通道贡献更多
    var_norm = _norm_ch(ch_var).squeeze(-1).squeeze(-1)  # [B,C]

    M_ch_mf = _norm_ch(ratio_mf * (1.0 + var_norm))
    M_ch_hf = _norm_ch(ratio_hf)
    M_ch_lf = _norm_ch(ratio_lf)

    return {'mf': M_ch_mf, 'hf': M_ch_hf, 'lf': M_ch_lf}


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def apply_lfim_v2(
    z:              torch.Tensor,        # refined latent [B,C,H,W]
    lq_image:       torch.Tensor = None, # LQ图像 [B,3,H,W]，推荐提供
    lf_alpha:       float = 0.0,         # 低频注入强度
    mf_beta:        float = 0.3,         # 中频注入强度（主要调节项）
    hf_beta:        float = 0.1,         # 高频注入强度
    cutoff_low:     float = 0.1,         # 低/中频分界
    cutoff_mid:     float = 0.3,         # 中/高频分界
    butter_order:   int   = 2,           # Butterworth 阶数
    entropy_weight: float = 0.5,         # 局部熵在中频空间门控中的权重
) -> torch.Tensor:
    """
    三频段自适应潜在频率注入。

    核心改进（相比 FiDeSR LFIM）：
    1. 三频段分解，中频单独处理
    2. 中频空间门控 = Sobel（边缘）+ 局部熵（纹理），互补覆盖细节区域
    3. 高频空间门控只用 Sobel，更保守避免噪声
    4. 通道门控结合频率能量比和通道激活方差

    参数：
        lf_alpha      : 低频注入强度（默认 0，可从 0.1 开始试）
        mf_beta       : 中频注入强度（主要调节，推荐 0.1~0.5）
        hf_beta       : 高频注入强度（推荐 0.05~0.2，不宜过大）
        cutoff_low    : 低/中频分界（默认 0.1）
        cutoff_mid    : 中/高频分界（默认 0.3）
        entropy_weight: 局部熵权重（0=纯Sobel，1=纯熵，默认0.5）
    """
    if lf_alpha == 0.0 and mf_beta == 0.0 and hf_beta == 0.0:
        return z

    orig_dtype = z.dtype
    B, C, H, W = z.shape
    z_f32 = z.float()

    # ── 1. 三频段分解（FFT）────────────────────────────────────────────────
    Z = torch.fft.fft2(z_f32)                          # [B,C,H,W] 复数

    lp_low_mask = _butterworth_lp(
        H, W, cutoff_low, butter_order, z.device).view(1,1,H,W)
    bp_mask = _bandpass_mask(
        H, W, cutoff_low, cutoff_mid, butter_order, z.device).view(1,1,H,W)
    hp_mask = _highpass_mask(
        H, W, cutoff_mid, butter_order, z.device).view(1,1,H,W)

    delta_lf = torch.fft.ifft2(Z * lp_low_mask).real   # 低频分量
    delta_mf = torch.fft.ifft2(Z * bp_mask).real        # 中频分量
    delta_hf = torch.fft.ifft2(Z * hp_mask).real        # 高频分量

    # ── 2. 空间门控（从 LQ 图或 latent 提取）──────────────────────────────
    if lq_image is not None:
        # 优先用 LQ 图（像素空间细节更清晰）
        # 下采样到 latent 尺寸
        src = F.interpolate(lq_image.float(), size=(H, W),
                            mode='bilinear', align_corners=False)
    else:
        src = z_f32

    sp_gates = _build_spatial_gates(src, entropy_weight)
    M_sp_lf  = sp_gates['lf']                          # [B,1,H,W]
    M_sp_mf  = sp_gates['mf']                          # [B,1,H,W]
    M_sp_hf  = sp_gates['hf']                          # [B,1,H,W]

    # ── 3. 通道门控 ────────────────────────────────────────────────────────
    ch_gates = _channel_gates(z_f32, cutoff_low, cutoff_mid)
    M_ch_lf  = ch_gates['lf']                          # [B,C,1,1]
    M_ch_mf  = ch_gates['mf']                          # [B,C,1,1]
    M_ch_hf  = ch_gates['hf']                          # [B,C,1,1]

    # ── 4. 注入 ───────────────────────────────────────────────────────────
    z_out = z_f32.clone()

    if lf_alpha > 0.0:
        # 低频注入：平坦区域，稳定结构和色调
        z_out = z_out + lf_alpha * M_sp_lf * M_ch_lf * delta_lf

    if mf_beta > 0.0:
        # 中频注入：纹理+边缘区域，核心感知质量提升
        z_out = z_out + mf_beta * M_sp_mf * M_ch_mf * delta_mf

    if hf_beta > 0.0:
        # 高频注入：边缘区域，锐化细节（保守）
        z_out = z_out + hf_beta * M_sp_hf * M_ch_hf * delta_hf

    return z_out.to(orig_dtype)