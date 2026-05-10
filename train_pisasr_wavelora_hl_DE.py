"""
train_pisasr_wavelora_hl_DE.py

在 train_pisasr_WaveLoRA.py 基础上新增：
  - DAW（Detail-aware Weighting）loss 指导
      Stage1：只对 loss_l2 加权（lambda_lpips=0，lambda_csd=0）
      Stage2：对 loss_l2 + loss_lpips + loss_csd 同时加权

DAW 逻辑（对应 FiDeSR Algorithm 1）：
  D  = Sobel + Laplacian + LocalVariance（从 GT x_tgt 提取）
  E  = L1(x_tgt_pred, x_tgt)（预测误差图，需要 GT）
  W  = tanh(blur(D ⊙ E))
  w* = mean_norm(1 + alpha * W)   均值≈1，loss 量级不变

其余逻辑与 train_pisasr_WaveLoRA.py 完全一致。
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

from pisasr_wave_hl import CSDLoss, PiSASR
from src.my_utils.training_utils import parse_args
from src.datasets.dataset import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import random


# ══════════════════════════════════════════════════════════════════════════════
# DAW 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _build_daw_kernels(device):
    """构造 DAW 所需的固定卷积核，返回 dict。每次调用会检查 device。"""
    kx  = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    ).view(1, 1, 3, 3).to(device)
    ky  = kx.transpose(-1, -2).contiguous()
    lap = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
    ).view(1, 1, 3, 3).to(device)
    k3  = torch.ones(1, 1, 3, 3).to(device) / 9.0
    return dict(kx=kx, ky=ky, lap=lap, k3=k3)


@torch.no_grad()
def compute_daw_weights(
    x_pred: torch.Tensor,   # 预测图 [B,3,H,W], float32, [-1,1]
    x_gt:   torch.Tensor,   # GT 图  [B,3,H,W], float32, [-1,1]
    kernels: dict,
    alpha:   float = 2.0,
    q:       float = 0.99,
) -> torch.Tensor:
    """
    计算 DAW 难度权重图 w_star [B,1,H,W]，均值≈1。

    D（细节丰富度，来自 GT）：
        Sobel + Laplacian + LocalVariance 均值，quantile 归一化后 blur。

    E（预测误差，来自 pred vs GT）：
        逐像素 L1 误差，quantile 归一化。
        （不用 LPIPS 是因为 LPIPS 默认返回标量，空间分辨率不足；
          如需精确感知误差，可把 lpips spatial=True 的结果加权混合。）

    W_DAW = D ⊙ E → blur → tanh
    w*    = mean_norm(1 + alpha * W_DAW)
    """
    kx, ky, lap, k3 = kernels['kx'], kernels['ky'], kernels['lap'], kernels['k3']
    B = x_gt.shape[0]

    # ── D：从 GT 提取细节图 ─────────────────────────────────────────────────
    # ★ 显式 .float() 确保 float32，quantile() 不支持 fp16/bf16
    y_gray = (0.299 * x_gt[:, 0:1]
            + 0.587 * x_gt[:, 1:2]
            + 0.114 * x_gt[:, 2:3]).float()                  # [B,1,H,W] float32

    # kernel 也强制 float32
    kx  = kx.float()
    ky  = ky.float()
    lap = lap.float()
    k3  = k3.float()

    gx    = F.conv2d(y_gray, kx,  padding=1)
    gy    = F.conv2d(y_gray, ky,  padding=1)
    sobel = (gx**2 + gy**2 + 1e-8).sqrt()

    laplacian = F.conv2d(y_gray, lap, padding=1).abs()

    mu   = F.avg_pool2d(y_gray,    7, stride=1, padding=3)
    mu2  = F.avg_pool2d(y_gray**2, 7, stride=1, padding=3)
    var  = (mu2 - mu**2).clamp(min=0)

    D = (sobel + laplacian + var) / 3.0
    # ★ D 已是 float32（y_gray 是 float32），quantile 可正常执行
    D_q = D.reshape(B, -1).quantile(q, dim=1).clamp(min=1e-6).view(B, 1, 1, 1)
    D = (D / D_q).clamp(0, 1)
    D = F.conv2d(D, k3, padding=1)                           # 3×3 blur

    # ── E：预测误差图（L1）──────────────────────────────────────────────────
    # ★ 显式 .float() 确保 fp32，防止 fp16 输入导致 quantile 报错
    E = (x_pred.float() - x_gt.float()).abs().mean(dim=1, keepdim=True)
    E_q = E.reshape(B, -1).quantile(q, dim=1).clamp(min=1e-6).view(B, 1, 1, 1)
    E = (E / E_q).clamp(0, 1)

    # ── W_DAW = D ⊙ E ───────────────────────────────────────────────────────
    W = torch.tanh(F.conv2d(D * E, k3, padding=1))           # tanh 防止极端值

    # ── w* = mean_norm(1 + alpha * W) ───────────────────────────────────────
    raw    = 1.0 + alpha * W                                  # [B,1,H,W]
    mean_  = raw.reshape(B, -1).mean(dim=1).view(B, 1, 1, 1).clamp(min=1e-6)
    w_star = raw / mean_                                      # 均值归一化

    return w_star                                             # [B,1,H,W]


def daw_weighted_l2(x_pred, x_gt, w_star):
    """DAW 加权 MSE loss。"""
    return (w_star * (x_pred.float() - x_gt.float())**2).mean()


def daw_weighted_lpips(x_pred, x_gt, w_star, net_lpips):
    """
    DAW 加权 LPIPS loss。
    net_lpips(a, b) 默认返回 [B] 标量，
    用 w_star 的 per-sample 均值近似空间加权。
    """
    lpips_val = net_lpips(x_pred.float(), x_gt.float())      # [B] 或 [B,1,1,1]
    if lpips_val.dim() == 1:
        # 标量版：用每个样本的 w_star 均值缩放
        w_per_sample = w_star.reshape(w_star.shape[0], -1).mean(dim=1)  # [B]
        return (w_per_sample * lpips_val).mean()
    else:
        # 空间版（spatial=True）：直接乘
        w_r = F.interpolate(w_star, size=lpips_val.shape[-2:],
                            mode='bilinear', align_corners=False)
        return (w_r * lpips_val).mean()


def daw_weighted_csd(latents, prompt_embeds, neg_prompt_embeds,
                     args, net_csd, w_star):
    """
    DAW 加权 CSD loss（近似版）。
    cal_csd 返回标量，用 latent 尺寸的 w_star 均值近似加权。
    """
    csd_val = net_csd.cal_csd(latents, prompt_embeds, neg_prompt_embeds, args)
    # w_star 是像素空间，latent 是 1/8 分辨率
    _, _, lh, lw = latents.shape
    w_lat = F.interpolate(w_star, size=(lh, lw),
                          mode='bilinear', align_corners=False)
    w_scalar = w_lat.reshape(w_lat.shape[0], -1).mean(dim=1).mean()
    return csd_val * w_scalar


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── Stage1 初始设置 ──────────────────────────────────────────────────────
    net_pisasr.unet.set_adapter(
        ['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'])
    net_pisasr.set_train_pix()

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

    # DAW 超参（可以加到 args 里，这里先给默认值）
    daw_alpha     = getattr(args, 'daw_alpha', 2.0)   # w* 放大系数
    daw_q         = getattr(args, 'daw_q',     0.99)  # quantile 归一化分位数

    # DAW 卷积核（懒初始化，第一个 batch 时构造）
    daw_kernels = None

    if args.resume_ckpt is not None:
        args.pix_steps = 1

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_pisasr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                # ── RAM caption ───────────────────────────────────────────────
                x_tgt_ram = ram_transforms(x_tgt * 0.5 + 0.5)
                with torch.no_grad():
                    caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{c}, {args.pos_prompt_csd}' for c in caption]

                # ── Stage2 切换 ───────────────────────────────────────────────
                if global_step == args.pix_steps:
                    if args.is_module:
                        net_pisasr.module.unet.set_adapter([
                            'default_encoder_pix', 'default_decoder_pix',
                            'default_others_pix',
                            'default_encoder_sem', 'default_decoder_sem',
                            'default_others_sem'])
                        net_pisasr.module.set_train_sem()
                    else:
                        net_pisasr.unet.set_adapter([
                            'default_encoder_pix', 'default_decoder_pix',
                            'default_others_pix',
                            'default_encoder_sem', 'default_decoder_sem',
                            'default_others_sem'])
                        net_pisasr.set_train_sem()

                    layers_to_opt.clear()
                    for n, _p in accelerator.unwrap_model(net_pisasr).unet.named_parameters():
                        if "lora" in n and "sem" in n:
                            layers_to_opt.append(_p)
                    layers_to_opt.extend(
                        accelerator.unwrap_model(net_pisasr).get_sem_wave_params())
                    optimizer.param_groups[0]['params'] = layers_to_opt

                    stage2_lr = args.learning_rate * 1
                    for pg in optimizer.param_groups:
                        pg['lr'] = stage2_lr
                    print(f"[Stage2] lr = {stage2_lr}")

                    lambda_l2    = args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_csd   = args.lambda_csd

                # ── forward ──────────────────────────────────────────────────
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = \
                    net_pisasr(x_src, x_tgt, batch=batch, args=args)

                # ── loss 计算 ────────────────────────────────────────────────
                # Stage1：原版 loss，不加 DAW
                # Stage2：完整 D×E DAW 加权三个 loss
                if global_step < args.pix_steps:
                    # Stage1：原版，不加 DAW
                    loss_l2    = F.mse_loss(
                        x_tgt_pred.float(), x_tgt.float(), reduction="mean"
                    ) * lambda_l2
                    loss_lpips = torch.tensor(0.0, device=x_tgt.device)
                    loss_csd   = torch.tensor(0.0, device=x_tgt.device)
                    w_star_mean = torch.tensor(1.0)
                    w_star_std  = torch.tensor(0.0)

                else:
                    # Stage2：完整 D×E DAW 加权
                    if daw_kernels is None:
                        daw_kernels = _build_daw_kernels(x_tgt.device)

                    w_star = compute_daw_weights(
                        x_pred  = x_tgt_pred.detach(),
                        x_gt    = x_tgt,
                        kernels = daw_kernels,
                        alpha   = daw_alpha,
                        q       = daw_q,
                    )                                         # [B,1,H,W]

                    loss_l2    = daw_weighted_l2(
                        x_tgt_pred.float(), x_tgt.float(), w_star
                    ) * lambda_l2

                    loss_lpips = daw_weighted_lpips(
                        x_tgt_pred.float(), x_tgt.float(), w_star, net_lpips
                    ) * lambda_lpips

                    loss_csd   = daw_weighted_csd(
                        latents_pred, prompt_embeds, neg_prompt_embeds,
                        args, net_csd, w_star
                    ) * lambda_csd

                    w_star_mean = w_star.mean()
                    w_star_std  = w_star.std()

                loss = loss_l2 + loss_lpips + loss_csd

                # ── backward ──────────────────────────────────────────────────
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # ── logging & checkpoint ──────────────────────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_csd":   loss_csd.detach().item(),
                        "loss_l2":    loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        # Stage1 时均值=1，std=0（占位）；Stage2 时为真实 DAW 统计
                        "daw_w_mean": w_star_mean.item(),
                        "daw_w_std":  w_star_std.item(),
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(
                            args.output_dir, "checkpoints",
                            f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pisasr).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(
                            args.output_dir, "eval", f"fid_{global_step}"),
                            exist_ok=True)
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
                                outf = os.path.join(
                                    args.output_dir, "eval",
                                    f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args()
    main(args)


'''
tmux new -s pisasr_wavelora_hl_DE -d "CUDA_VISIBLE_DEVICES=\"6,7\" accelerate launch \
--main_process_port 22869 \
--num_processes 2 \
train_pisasr_wavelora_hl_DE.py \
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
--output_dir=experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-DE \
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
--tracker_project_name=PiSASR-wavelora-hl-DE \
--mixed_precision=fp16 \
--max_train_steps=20000 \
--is_module=True \
2>&1 | tee pisasr_wavelora_hl_DE.log"
'''