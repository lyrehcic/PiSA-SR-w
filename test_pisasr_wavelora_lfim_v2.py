"""
test_pisasr_wavelora_lfim_v2.py

在 test_pisasr_wavelora.py 基础上加入三频段 LFIM v2 推理时注入。
不需要重训练，直接用已有 checkpoint 测试。

新增参数：
    --lf_alpha      低频注入强度（默认 0.0）
    --mf_beta       中频注入强度（默认 0.3，主要调节项）
    --hf_beta       高频注入强度（默认 0.1）
    --cutoff_low    低/中频分界（默认 0.1）
    --cutoff_mid    中/高频分界（默认 0.3）
    --entropy_weight 局部熵权重（默认 0.5）
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
import time
from torchvision import transforms
import torchvision.transforms.functional as F

from pisasr_wave_hl import PiSASR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from lfim_v2 import apply_lfim_v2

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
    print(f'LFIM v2: lf_alpha={args.lf_alpha}, mf_beta={args.mf_beta}, '
          f'hf_beta={args.hf_beta}, entropy_weight={args.entropy_weight}')

    time_records = []
    for image_name in image_names:
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or \
           ori_height < args.process_size // rscale:
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
            c_t_dtype     = c_t.to(dtype=model.weight_dtype)
            prompt_embeds = model.encode_prompt([validation_prompt]).to(
                dtype=model.weight_dtype)
            encoded_control = (model.vae.encode(c_t_dtype).latent_dist.sample()
                               * model.vae.config.scaling_factor)

            torch.cuda.synchronize()
            t0 = time.time()

            # U-Net 推理
            model_pred = model._process_latents(
                encoded_control, prompt_embeds, args.default)
            x_denoised = encoded_control - model_pred   # refined latent z_r

            # ── LFIM v2：三频段自适应注入 ──────────────────────────────────
            use_lfim = (args.lf_alpha > 0.0 or
                        args.mf_beta  > 0.0 or
                        args.hf_beta  > 0.0)
            if use_lfim:
                x_denoised = apply_lfim_v2(
                    z              = x_denoised,
                    lq_image       = c_t,
                    lf_alpha       = args.lf_alpha,
                    mf_beta        = args.mf_beta,
                    hf_beta        = args.hf_beta,
                    cutoff_low     = args.cutoff_low,
                    cutoff_mid     = args.cutoff_mid,
                    entropy_weight = args.entropy_weight,
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
                (int(args.upscale * ori_width),
                 int(args.upscale * ori_height)))
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

    # ── LFIM v2 参数 ───────────────────────────────────────────────────────
    parser.add_argument("--lf_alpha", type=float, default=0.0,
                        help="低频注入强度，推荐 0.0~0.2")
    parser.add_argument("--mf_beta", type=float, default=0.0,
                        help="中频注入强度（主要调节项），推荐 0.1~0.5")
    parser.add_argument("--hf_beta", type=float, default=0.0,
                        help="高频注入强度，推荐 0.05~0.2")
    parser.add_argument("--cutoff_low", type=float, default=0.1,
                        help="低/中频分界截止频率")
    parser.add_argument("--cutoff_mid", type=float, default=0.3,
                        help="中/高频分界截止频率")
    parser.add_argument("--entropy_weight", type=float, default=0.5,
                        help="局部熵在中频空间门控中的权重，0=纯Sobel，1=纯熵")

    args = parser.parse_args()
    pisa_sr(args)


"""
# ══════════════════════════════════════════════════════════════════════════════
# 推荐测试命令
# ══════════════════════════════════════════════════════════════════════════════

CKPT=/data/wyb/PiSA-SR/experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-lr1_linear_66M/checkpoints/model_14501.pkl
SD=/data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base
LR=/data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/test_LR
OUT=/data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/lfim_v2

# Step1：先扫描 mf_beta（固定 lf=0, hf=0.1）
for mf in 0.0 0.1 0.2 0.3 0.4 0.5; do
python test_pisasr_wavelora_lfim_v2.py \
  --pretrained_model_path ${SD} --pretrained_path ${CKPT} \
  --process_size 512 --upscale 4 --default \
  --input_image ${LR} \
  --output_dir ${OUT}/mf${mf}_hf0.1 \
  --mf_beta ${mf} --hf_beta 0.1 --lf_alpha 0.0
done

# Step2：找到最佳 mf_beta 后，扫描 entropy_weight（看 Sobel 和熵的比例）
for ew in 0.0 0.3 0.5 0.7 1.0; do
python test_pisasr_wavelora_lfim_v2.py \
  --pretrained_model_path ${SD} --pretrained_path ${CKPT} \
  --process_size 512 --upscale 4 --default \
  --input_image ${LR} \
  --output_dir ${OUT}/mf0.3_ew${ew} \
  --mf_beta 0.3 --hf_beta 0.1 --lf_alpha 0.0 \
  --entropy_weight ${ew}
done

# Step3：最佳组合加上 lf_alpha 看能否进一步提升 PSNR/SSIM
for lf in 0.0 0.1 0.2; do
python test_pisasr_wavelora_lfim_v2.py \
  --pretrained_model_path ${SD} --pretrained_path ${CKPT} \
  --process_size 512 --upscale 4 --default \
  --input_image ${LR} \
  --output_dir ${OUT}/mf0.3_hf0.1_lf${lf} \
  --mf_beta 0.3 --hf_beta 0.1 --lf_alpha ${lf}
done
"""