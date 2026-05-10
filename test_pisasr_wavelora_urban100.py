import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from pisasr_wave_hl import PiSASR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import glob


def pad_to_multiple_of_8(image: Image.Image):
    w, h = image.size
    new_w = (w + 7) // 8 * 8
    new_h = (h + 7) // 8 * 8
    if new_w == w and new_h == h:
        return image, w, h

    padded = Image.new("RGB", (new_w, new_h))
    padded.paste(image, (0, 0))

    if new_w > w:
        right_strip = image.crop((w - 1, 0, w, h)) \
                           .resize((new_w - w, h), Image.NEAREST)
        padded.paste(right_strip, (w, 0))
    if new_h > h:
        bottom_strip = image.crop((0, h - 1, new_w, h)) \
                            .resize((new_w, new_h - h), Image.NEAREST)
        padded.paste(bottom_strip, (0, h))

    return padded, w, h


def process_one_image(model, args, image_name):
    """
    单张图像推理，独立函数便于异常捕获和显存管理
    """
    bname = os.path.basename(image_name)

    lr_image = Image.open(image_name).convert('RGB')
    lr_w, lr_h = lr_image.size
    target_w = lr_w * args.upscale
    target_h = lr_h * args.upscale

    lr_padded, orig_lr_w, orig_lr_h = pad_to_multiple_of_8(lr_image)
    pad_lr_w, pad_lr_h = lr_padded.size

    hr_input = lr_padded.resize(
        (pad_lr_w * args.upscale, pad_lr_h * args.upscale),
        Image.BICUBIC
    )

    validation_prompt = ''

    with torch.no_grad():
        c_t = F.to_tensor(hr_input).unsqueeze(0).cuda() * 2 - 1
        inference_time, output_image = model(
            args.default, c_t, prompt=validation_prompt
        )
        # 后处理在 no_grad 内完成，减少显存占用
        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        # 立即转到 CPU，释放 GPU tensor
        output_pil = transforms.ToPILImage()(output_image[0].cpu())
        del output_image, c_t

    # 强制释放显存碎片
    torch.cuda.empty_cache()

    crop_hr_w = orig_lr_w * args.upscale
    crop_hr_h = orig_lr_h * args.upscale
    output_pil = output_pil.crop((0, 0, crop_hr_w, crop_hr_h))

    hr_input_cropped = hr_input.crop((0, 0, crop_hr_w, crop_hr_h))

    if args.align_method == 'adain':
        output_pil = adain_color_fix(target=output_pil, source=hr_input_cropped)
    elif args.align_method == 'wavelet':
        output_pil = wavelet_color_fix(target=output_pil, source=hr_input_cropped)

    assert output_pil.size == (target_w, target_h), \
        f"尺寸不匹配: {output_pil.size} vs ({target_w},{target_h})"

    return inference_time, output_pil, bname


def pisa_sr_urban100(args):
    model = PiSASR_eval(args)
    model.set_eval()

    if os.path.isdir(args.input_image):
        image_names = sorted(
            glob.glob(f'{args.input_image}/*.png') +
            glob.glob(f'{args.input_image}/*.jpg') +
            glob.glob(f'{args.input_image}/*.bmp')
        )
    else:
        image_names = [args.input_image]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'共找到 {len(image_names)} 张图像.')

    # ------------------------------------------------------------------ #
    # 断点续传：跳过已处理的图像
    # 方式1：--start_idx 指定从第几张开始（0-based）
    # 方式2：--skip_exist 自动跳过输出目录中已存在的文件
    # ------------------------------------------------------------------ #
    if args.skip_exist:
        before = len(image_names)
        image_names = [
            n for n in image_names
            if not os.path.exists(
                os.path.join(args.output_dir, os.path.basename(n))
            )
        ]
        print(f'跳过已存在的 {before - len(image_names)} 张，'
              f'剩余 {len(image_names)} 张.')
    elif args.start_idx > 0:
        print(f'从第 {args.start_idx} 张开始（跳过前 {args.start_idx} 张）.')
        image_names = image_names[args.start_idx:]

    if len(image_names) == 0:
        print('所有图像已处理完毕，退出.')
        return

    time_records = []
    failed = []

    for i, image_name in enumerate(image_names):
        bname = os.path.basename(image_name)
        try:
            inference_time, output_pil, bname = process_one_image(
                model, args, image_name
            )
            output_pil.save(os.path.join(args.output_dir, bname))
            print(f"[{i+1}/{len(image_names)}] {bname}  耗时:{inference_time:.4f}s  "
                  f"显存:{torch.cuda.memory_allocated()/1024**3:.2f}GB")
            time_records.append(inference_time)

        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] {bname} 显存不足，尝试清理后跳过: {e}")
            torch.cuda.empty_cache()
            failed.append(image_name)
            continue

        except Exception as e:
            print(f"[ERROR] {bname} 处理失败: {e}")
            failed.append(image_name)
            continue

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records) if time_records else 0.0
    print(f"\n平均推理时间（跳过前3张）: {average_time:.4f} 秒")

    if failed:
        print(f"\n以下 {len(failed)} 张处理失败：")
        for f in failed:
            print(f"  {f}")


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
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str,
                        choices=['wavelet', 'adain', 'nofix'],
                        default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float)
    parser.add_argument("--lambda_sem", default=1.0, type=float)
    parser.add_argument("--wave_scale", default=0.2, type=float)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp32")
    parser.add_argument("--default", action="store_true")
    parser.add_argument("--process_size", type=int, default=512)
    # ------------------------------------------------------------------ #
    # 新增：断点续传参数
    # ------------------------------------------------------------------ #
    parser.add_argument("--start_idx", type=int, default=0,
                        help="从第几张图开始处理（0-based），用于手动指定断点")
    parser.add_argument("--skip_exist", action="store_true",
                        help="自动跳过输出目录中已存在的文件（推荐）")
    args = parser.parse_args()

    pisa_sr_urban100(args)
    
"""

python test_pisasr_wavelora_urban100.py \
--pretrained_model_path /data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base \
--pretrained_path /data/wyb/PiSA-SR/preset/models/pisa_sr.pkl \
--upscale 4 \
--input_image /data/wyb/PiSA-SR/Urban100/LR \
--output_dir /data/wyb/PiSA-SR/Urban100/results_scale1_wave_hl_AB_9001 \
--align_method adain \
--default

python test_pisasr_wavelora_urban100.py \
--pretrained_model_path /data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base \
--pretrained_path /data/wyb/PiSA-SR/experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-lr1_linear_scale1/checkpoints/model_9001.pkl \
--upscale 4 \
--input_image /data/wyb/PiSA-SR/Urban100/LR \
--output_dir /data/wyb/PiSA-SR/Urban100/results_scale1_wave_hl_AB_9001 \
--align_method adain \
--latent_tiled_size 200 \
--latent_tiled_overlap 32 \
--default


python test_metrics.py \
--inp_imgs /data/wyb/PiSA-SR/Urban100/results_scale1_wave_hl_AB_9001 \
--gt_imgs /data/wyb/PiSA-SR/Urban100/HR \
--log /data/wyb/PiSA-SR/Urban100/results_scale1_wave_hl_AB_9001/metrics



    """
