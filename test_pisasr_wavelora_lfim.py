"""
test_pisasr_wavelora_lfim.py

在 test_pisasr_wavelora.py 基础上加入 LFIM 推理时高低频注入。
不需要重训练，直接用已有 checkpoint 测试不同 hf_beta 的效果。

新增参数：
    --hf_beta   高频注入强度（默认 0.0 = 不注入，推荐测试 0.1~0.5）
    --lf_alpha  低频注入强度（默认 0.0 = 不注入，推荐测试 0.0~0.3）
    --hf_cutoff 高通截止频率比例（默认 0.3）
    --lf_cutoff 低通截止频率比例（默认 0.1）
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from pisasr_wave_hl import PiSASR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from lfim import apply_lfim

import glob


def pisa_sr(args):
    model = PiSASR_eval(args)
    model.set_eval()

    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')
    print(f'LFIM: hf_beta={args.hf_beta}, lf_alpha={args.lf_alpha}')

    time_records = []
    for image_name in image_names:
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize(
                (int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize(
            (input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width  = input_image.width  - input_image.width  % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        validation_prompt = ''

        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1

            # ── 原版推理到 latent ──────────────────────────────────────────
            # 复用 model 内部的 encode + unet + latent，
            # 在 VAE decode 之前插入 LFIM
            c_t_dtype = c_t.to(dtype=model.weight_dtype)
            prompt_embeds = model.encode_prompt([validation_prompt]).to(
                dtype=model.weight_dtype)
            encoded_control = (model.vae.encode(c_t_dtype).latent_dist.sample()
                               * model.vae.config.scaling_factor)

            # U-Net 预测（复用 _process_latents 逻辑）
            torch.cuda.synchronize()
            import time
            t0 = time.time()

            model_pred = model._process_latents(
                encoded_control, prompt_embeds, args.default)
            x_denoised = encoded_control - model_pred   # refined latent z_r

            # ── LFIM：在 VAE decode 前注入高低频 ──────────────────────────
            if args.hf_beta > 0.0 or args.lf_alpha > 0.0:
                x_denoised = apply_lfim(
                    z         = x_denoised,
                    hf_beta   = args.hf_beta,
                    lf_alpha  = args.lf_alpha,
                    lq_image  = c_t,              # 用 LQ 图像做空间门控
                    hf_cutoff = args.hf_cutoff,
                    lf_cutoff = args.lf_cutoff,
                )

            # VAE decode
            output_image = model.vae.decode(
                x_denoised / model.vae.config.scaling_factor
            ).sample.clamp(-1, 1)

            torch.cuda.synchronize()
            inference_time = time.time() - t0

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil   = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize(
                (int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.output_dir, bname))

    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str,
                        default='preset/test_datasets')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='experiments/test')
    parser.add_argument("--pretrained_model_path", type=str,
                        default='preset/models/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str,
                        default='preset/models/pisa_sr.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str,
                        choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float)
    parser.add_argument("--lambda_sem", default=1.0, type=float)
    parser.add_argument("--wave_scale", default=0.2, type=float)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp32")
    parser.add_argument("--default", action="store_true")

    # ── LFIM 参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--hf_beta", type=float, default=0.0,
                        help="高频注入强度，0=不注入，推荐测试 0.1/0.2/0.3/0.4/0.5")
    parser.add_argument("--lf_alpha", type=float, default=0.0,
                        help="低频注入强度，0=不注入，推荐测试 0.0/0.1/0.2/0.3")
    parser.add_argument("--hf_cutoff", type=float, default=0.3,
                        help="高通截止频率比例，越小高频越多")
    parser.add_argument("--lf_cutoff", type=float, default=0.1,
                        help="低通截止频率比例，越大低频越多")

    args = parser.parse_args()
    pisa_sr(args)


"""
# ── 推荐测试命令 ──────────────────────────────────────────────────────────

BASE="python test_pisasr_wavelora_lfim.py \
--pretrained_model_path /data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base \
--pretrained_path /data/wyb/PiSA-SR/experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-DE/checkpoints/model_8001.pkl \
--process_size 512 --upscale 4 --default \
--input_image /data/wyb/PiSA-SR/preset/datasets/benchmark_drealsr/test_LR"

# baseline（不注入）
python test_pisasr_wavelora_lfim.py ... --hf_beta 0.0 --output_dir results_lfim_hf0.0

# 扫描 hf_beta（只注入高频）
python test_pisasr_wavelora_lfim.py ... --hf_beta 0.1 --output_dir results_lfim_hf0.1
python test_pisasr_wavelora_lfim.py ... --hf_beta 0.2 --output_dir results_lfim_hf0.2
python test_pisasr_wavelora_lfim.py ... --hf_beta 0.3 --output_dir results_lfim_hf0.3
python test_pisasr_wavelora_lfim.py ... --hf_beta 0.4 --output_dir results_lfim_hf0.4
python test_pisasr_wavelora_lfim.py ... --hf_beta 0.5 --output_dir results_lfim_hf0.5

# 找到最佳 hf_beta 后，再测 lf_alpha 的效果
python test_pisasr_wavelora_lfim.py ... --hf_beta 0.3 --lf_alpha 0.1 --output_dir results_lfim_hf0.3_lf0.1
python test_pisasr_wavelora_lfim.py ... --hf_beta 0.3 --lf_alpha 0.2 --output_dir results_lfim_hf0.3_lf0.2
"""
