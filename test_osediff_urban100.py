import os
import sys
sys.path.append("/data/wyb/OSEDiff")
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from osediff_copy import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def pad_to_multiple_of_8(image: Image.Image):
    """右/下 pad 到 8 的倍数，用边缘像素填充"""
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


def get_validation_prompt(args, image, model, weight_dtype, device='cuda'):
    """用 RAM 生成 caption"""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"
    return validation_prompt, lq


def process_one_image(args, model, DAPE, weight_dtype, image_name):
    """
    单张图像完整推理流程（Urban100 版本）
    输入：LR 图像
    输出：HR 结果 PIL image
    """
    # ------------------------------------------------------------------ #
    # 1. 读取 LR 图像
    # ------------------------------------------------------------------ #
    lr_image = Image.open(image_name).convert('RGB')
    lr_w, lr_h = lr_image.size

    # 期望输出尺寸
    target_w = lr_w * args.upscale
    target_h = lr_h * args.upscale

    # ------------------------------------------------------------------ #
    # 2. LR → bicubic ×upscale → HR 空间
    #    OSEDiff 的输入就是 bicubic 放大后的图
    # ------------------------------------------------------------------ #
    hr_bicubic = lr_image.resize(
        (lr_w * args.upscale, lr_h * args.upscale),
        Image.BICUBIC
    )

    # ------------------------------------------------------------------ #
    # 3. Pad 到 8 的倍数（在 HR 空间操作）
    # ------------------------------------------------------------------ #
    hr_padded, orig_hr_w, orig_hr_h = pad_to_multiple_of_8(hr_bicubic)
    pad_hr_w, pad_hr_h = hr_padded.size

    # ------------------------------------------------------------------ #
    # 4. 生成 RAM caption（用 padded 图，与推理输入一致）
    # ------------------------------------------------------------------ #
    validation_prompt, lq = get_validation_prompt(
        args, hr_padded, DAPE, weight_dtype
    )

    # ------------------------------------------------------------------ #
    # 5. 推理
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        lq = lq * 2 - 1  # [0,1] → [-1,1]
        output_image = model(lq, prompt=validation_prompt)
        output_image = output_image[0].cpu() * 0.5 + 0.5   # [-1,1]→[0,1]
        output_image = torch.clamp(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image)
        del output_image, lq

    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 6. 裁剪掉 padding（还原到真实 HR 尺寸）
    # ------------------------------------------------------------------ #
    output_pil = output_pil.crop((0, 0, orig_hr_w, orig_hr_h))
    # 同步裁剪 bicubic 参考图（用于颜色对齐）
    hr_bicubic_cropped = hr_bicubic.crop((0, 0, orig_hr_w, orig_hr_h))

    # ------------------------------------------------------------------ #
    # 7. 颜色对齐
    # ------------------------------------------------------------------ #
    if args.align_method == 'adain':
        output_pil = adain_color_fix(
            target=output_pil, source=hr_bicubic_cropped)
    elif args.align_method == 'wavelet':
        output_pil = wavelet_color_fix(
            target=output_pil, source=hr_bicubic_cropped)

    # ------------------------------------------------------------------ #
    # 8. 安全检查
    # ------------------------------------------------------------------ #
    assert output_pil.size == (target_w, target_h), \
        f"尺寸不匹配: {output_pil.size} vs ({target_w},{target_h})"

    return output_pil, validation_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str,
                        default='preset/datasets/test_dataset/input')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='preset/datasets/test_dataset/output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str,
                        choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--osediff_path", type=str,
                        default='preset/models/osediff.pkl')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision
    parser.add_argument("--mixed_precision", type=str,
                        choices=['fp16', 'fp32'], default="fp16")
    # lora
    parser.add_argument("--merge_and_unload_lora", default=False)
    # tile
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    # process_size 保留兼容性，Urban100 不使用
    parser.add_argument("--process_size", type=int, default=512)
    # 断点续传
    parser.add_argument("--start_idx", type=int, default=0,
                        help="从第几张开始处理（0-based）")
    parser.add_argument("--skip_exist", action="store_true",
                        help="自动跳过已存在的输出文件")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 初始化模型
    # ------------------------------------------------------------------ #
    model = OSEDiff_test(args)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # RAM 模型
    DAPE = ram(
        pretrained=args.ram_path,
        pretrained_condition=args.ram_ft_path,
        image_size=384,
        vit='swin_l'
    )
    DAPE.eval()
    DAPE.to("cuda", dtype=weight_dtype)

    # ------------------------------------------------------------------ #
    # 获取图像列表
    # ------------------------------------------------------------------ #
    if os.path.isdir(args.input_image):
        image_names = sorted(
            glob.glob(f'{args.input_image}/*.png') +
            glob.glob(f'{args.input_image}/*.jpg') +
            glob.glob(f'{args.input_image}/*.bmp')
        )
    else:
        image_names = [args.input_image]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)

    print(f'共找到 {len(image_names)} 张图像.')

    # ------------------------------------------------------------------ #
    # 断点续传
    # ------------------------------------------------------------------ #
    if args.skip_exist:
        before = len(image_names)
        image_names = [
            n for n in image_names
            if not os.path.exists(
                os.path.join(args.output_dir, os.path.basename(n))
            )
        ]
        print(f'跳过已存在 {before - len(image_names)} 张，'
              f'剩余 {len(image_names)} 张.')
    elif args.start_idx > 0:
        print(f'从第 {args.start_idx} 张开始（跳过前 {args.start_idx} 张）.')
        image_names = image_names[args.start_idx:]

    if not image_names:
        print('所有图像已处理完毕，退出.')
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    failed = []
    for i, image_name in enumerate(image_names):
        bname = os.path.basename(image_name)
        try:
            output_pil, validation_prompt = process_one_image(
                args, model, DAPE, weight_dtype, image_name
            )

            # 保存结果
            output_pil.save(os.path.join(args.output_dir, bname))

            # 保存 prompt
            if args.save_prompts:
                txt_save_path = os.path.join(
                    txt_path, bname.rsplit('.', 1)[0] + '.txt'
                )
                with open(txt_save_path, 'w', encoding='utf-8') as f:
                    f.write(validation_prompt)

            print(f"[{i+1}/{len(image_names)}] {bname}  "
                  f"prompt: {validation_prompt[:60]}...  "
                  f"显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] {bname} 显存不足，跳过: {e}")
            torch.cuda.empty_cache()
            failed.append(image_name)

        except Exception as e:
            print(f"[ERROR] {bname} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            failed.append(image_name)

    if failed:
        print(f"\n以下 {len(failed)} 张处理失败：")
        for f in failed:
            print(f"  {f}")
    else:
        print("\n所有图像处理完毕！")
        
"""
python test_osediff_urban100.py \
--pretrained_model_path /data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base \
--pretrained_path /data/wyb/OSEDiff/preset/models/osediff.pkl \
--upscale 4 \
--input_image /data/wyb/PiSA-SR/Urban100/LR \
--output_dir /data/wyb/PiSA-SR/Urban100/results_osediff \
--align_method adain \
--latent_tiled_size 200 \
--latent_tiled_overlap 32 \
--default


python test_metrics.py \
--inp_imgs /data/wyb/PiSA-SR/Urban100/results_osediff \
--gt_imgs /data/wyb/PiSA-SR/Urban100/HR \
--log /data/wyb/PiSA-SR/Urban100/results_osediff/metrics


"""
