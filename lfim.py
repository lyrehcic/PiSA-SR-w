"""
lfim.py

Latent Frequency Injection Module（推理时高频注入）
参考 FiDeSR 论文 Section 3.6 实现。

完全无参数，推理时直接调用，不需要重训练。

用法：
    from lfim import apply_lfim
    z_enhanced = apply_lfim(z_r, lf_alpha=0.0, hf_beta=0.3,
                            lq_image=x_src)   # lq_image 可选

只调高频注入时：lf_alpha=0.0, hf_beta=0.1~0.5
同时调低高频时：lf_alpha=0.2, hf_beta=0.2（FiDeSR 默认）
"""

import torch
import torch.nn.functional as F


# ── Butterworth 滤波器 ─────────────────────────────────────────────────────

def _butterworth_mask(H: int, W: int, cutoff: float, order: int = 2,
                      device=None, dtype=torch.float32) -> torch.Tensor:
    """
    生成 Butterworth 低通滤波器 mask [H, W]，值域 [0,1]。
    高通 mask = 1 - 低通 mask。

    cutoff: 截止频率比例，0~1（相对于最大频率）
    order:  滤波器阶数，越大过渡越陡
    """
    # 频率坐标（中心化）
    fh = torch.fft.fftfreq(H, device=device).unsqueeze(1).expand(H, W)
    fw = torch.fft.fftfreq(W, device=device).unsqueeze(0).expand(H, W)
    freq = (fh**2 + fw**2).sqrt()                   # [H,W] 归一化频率

    # Butterworth 低通
    lp = 1.0 / (1.0 + (freq / (cutoff + 1e-8)) ** (2 * order))
    return lp.to(dtype)


# ── 空间门控（从 LQ 图或 latent 提取细节图）──────────────────────────────

@torch.no_grad()
def _spatial_gate(x: torch.Tensor, detail_thresh: float = 0.3) -> torch.Tensor:
    """
    从输入张量提取空间门控 M_sp [B,1,H,W]，值域 [0,1]。
    细节丰富区域（边缘/纹理）值高，平坦区域值低。

    x: [B,C,H,W] latent 或图像，float32
    """
    x_gray = x.mean(dim=1, keepdim=True).float()
    B = x_gray.shape[0]

    # Sobel
    kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],
                       device=x.device).view(1,1,3,3)
    ky = kx.transpose(-1,-2).contiguous()
    gx = F.conv2d(x_gray, kx, padding=1)
    gy = F.conv2d(x_gray, ky, padding=1)
    sobel = (gx**2 + gy**2 + 1e-8).sqrt()

    # 归一化
    q = sobel.reshape(B,-1).quantile(0.99, dim=1).clamp(1e-6).view(B,1,1,1)
    M_sp = (sobel / q).clamp(0, 1)

    return M_sp                                      # [B,1,H,W]


# ── 通道门控（频率能量比）────────────────────────────────────────────────

@torch.no_grad()
def _channel_gate_hf(z: torch.Tensor, cutoff: float = 0.3) -> torch.Tensor:
    """
    计算每个 latent 通道的高频能量占比，作为通道门控 M_ch [B,C,1,1]。
    高频能量占比高的通道 → 门控值高 → 高频注入更强。
    """
    B, C, H, W = z.shape
    z_f = torch.fft.fft2(z.float())                 # [B,C,H,W] 复数

    lp_mask = _butterworth_mask(H, W, cutoff,
                                device=z.device).unsqueeze(0).unsqueeze(0)
    hp_mask = 1.0 - lp_mask                         # [1,1,H,W]

    energy_total = (z_f.abs()**2).mean(dim=[-2,-1])  # [B,C]
    energy_hf    = (z_f.abs()**2 * hp_mask).mean(dim=[-2,-1])

    ratio = energy_hf / (energy_total + 1e-8)       # [B,C]
    # 归一化到 [0,1]
    ratio_min = ratio.min(dim=1, keepdim=True)[0]
    ratio_max = ratio.max(dim=1, keepdim=True)[0]
    M_ch = (ratio - ratio_min) / (ratio_max - ratio_min + 1e-8)  # [B,C]

    return M_ch.view(B, C, 1, 1)                    # [B,C,1,1]


# ── 主函数：apply_lfim ─────────────────────────────────────────────────────

@torch.no_grad()
def apply_lfim(
    z:          torch.Tensor,          # refined latent [B,C,H,W]
    hf_beta:    float = 0.2,           # 高频注入强度，0=不注入
    lf_alpha:   float = 0.0,           # 低频注入强度，默认不用
    lq_image:   torch.Tensor = None,   # LQ 图像 [B,3,H,W]，用于空间门控（可选）
    hf_cutoff:  float = 0.3,           # 高通截止频率比例
    lf_cutoff:  float = 0.1,           # 低通截止频率比例
    butter_order: int = 2,             # Butterworth 阶数
) -> torch.Tensor:
    """
    推理时高低频自适应注入（LFIM）。

    参数：
        z         : VAE decode 前的 refined latent [B,C,H,W]
        hf_beta   : 高频注入强度（推荐调节范围 0.1~0.5）
                    越大 → MANIQA/MUSIQ 越高，PSNR 略降
        lf_alpha  : 低频注入强度（推荐调节范围 0.0~0.3）
                    越大 → PSNR/SSIM 越高，感知指标略降
        lq_image  : 如果提供，用 LQ 图像计算空间门控（更准确）
                    如果不提供，用 z 本身计算（也可用）
        hf_cutoff : 高通截止频率，值越小高频越多
        lf_cutoff : 低通截止频率，值越大低频越多

    返回：
        z_enhanced [B,C,H,W]，与输入同 dtype/device
    """
    if hf_beta == 0.0 and lf_alpha == 0.0:
        return z                                     # 不注入，直接返回

    orig_dtype = z.dtype
    B, C, H, W = z.shape
    z_f32 = z.float()

    # ── FFT 分解 ──────────────────────────────────────────────────────────
    Z = torch.fft.fft2(z_f32)                        # [B,C,H,W] 复数

    lp_mask = _butterworth_mask(H, W, lf_cutoff, butter_order,
                                device=z.device).unsqueeze(0).unsqueeze(0)
    hp_mask = _butterworth_mask(H, W, hf_cutoff, butter_order,
                                device=z.device)
    hp_mask = (1.0 - hp_mask).unsqueeze(0).unsqueeze(0)  # 高通

    delta_lp = torch.fft.ifft2(Z * lp_mask).real    # 低频分量
    delta_hp = torch.fft.ifft2(Z * hp_mask).real    # 高频分量

    # ── 空间门控 M_sp ─────────────────────────────────────────────────────
    if lq_image is not None:
        # 优先用 LQ 图像（分辨率更高，细节更准确）
        # 需要把 LQ 图像下采样到 latent 尺寸
        lq_resized = F.interpolate(lq_image.float(), size=(H, W),
                                   mode='bilinear', align_corners=False)
        M_sp = _spatial_gate(lq_resized)             # [B,1,H,W]
    else:
        M_sp = _spatial_gate(z_f32)                  # [B,1,H,W]

    # ── 通道门控 M_ch ─────────────────────────────────────────────────────
    M_ch_hf = _channel_gate_hf(z_f32, cutoff=hf_cutoff)   # [B,C,1,1]

    # 低频通道门控：与高频互补
    M_ch_lf = 1.0 - M_ch_hf                               # [B,C,1,1]

    # ── 注入 ──────────────────────────────────────────────────────────────
    z_enhanced = z_f32.clone()

    if hf_beta > 0.0:
        # 高频注入：在细节丰富区域（M_sp高）、高频能量强的通道（M_ch_hf高）注入
        # 对应 FiDeSR: z ← z + hf_beta · M_sp^HF · M_ch^HF · ΔHP
        z_enhanced = z_enhanced + hf_beta * M_sp * M_ch_hf * delta_hp

    if lf_alpha > 0.0:
        # 低频注入：在平坦区域（1-M_sp）、低频能量强的通道注入
        # 对应 FiDeSR: z ← z + lf_alpha · M_sp · M_ch · ΔLP
        # 论文里低频注入在平坦区域限制，避免过度平滑细节区域
        lf_spatial = (1.0 - M_sp)                   # 平坦区域权重高
        z_enhanced = z_enhanced + lf_alpha * lf_spatial * M_ch_lf * delta_lp

    return z_enhanced.to(orig_dtype)
