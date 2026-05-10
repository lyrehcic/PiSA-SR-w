"""
test_pisasr_wavelora.py

WaveLoRA 版推理脚本，基于原版 test_pisasr.py。
支持 --lambda_pix, --lambda_sem 自定义调节。
支持 --wave_scale 控制 WaveAdapter 的 sem scale（默认 0.2）。
支持 --default 使用默认设置（不做差值，直接输出）。
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from pisasr_wave_hl_infer_wavescale_change import PiSASR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import glob


def pisa_sr(args):
    # Initialize the model
    model = PiSASR_eval(args)
    model.set_eval()

    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        # Ensure the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(args.default, c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.output_dir, bname))

    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets',
                        help="path to the input image or directory")
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test',
                        help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str,
                        default='preset/models/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str,
                        default='preset/models/pisa_sr.pkl',
                        help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str,
                        choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float,
                        help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float,
                        help="the scale for semantic-level enhancements")
    parser.add_argument("--wave_scale", default=0.2, type=float,
                        help="WaveAdapter sem scale used during inference")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp32")
    parser.add_argument("--default", action="store_true",
                        help="use default setting (no pix/sem separation)")

    args = parser.parse_args()

    # Call the processing function
    pisa_sr(args)
    
    
    
"""
python test_pisasr_wavelora.py \
--pretrained_model_path /data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base \
--pretrained_path /data/wyb/PiSA-SR/experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-lr1_linear_272M/checkpoints/model_18501.pkl \
--process_size 512 \
--upscale 4 \
--input_image /data/wyb/PiSA-SR/Urban100/LR \
--output_dir /data/wyb/PiSA-SR/Urban100/results_pisasr-wavelora-hl-lr1_linear_272M_18500 \
--default

python test_metrics.py \
--inp_imgs /data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/1_dataset/change_infer \
--gt_imgs /data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/test_HR \
--log /data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/1_dataset/change_infer/metrics



    """
