# Image Quality Assessment Script
# Evaluates metrics like PSNR, SSIM, LPIPS, FID, DISTS, etc., for a set of images.

import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import cv2
import numpy as np
import torch

# ========== 核心配置：指定权重路径 ==========
CUSTOM_TORCH_HOME = '/hy-tmp/checkpoints/OSEDiff/outcome_index'
os.environ['TORCH_HOME'] = CUSTOM_TORCH_HOME
# ★ 删除 PYIQA_DOWNLOAD=False，允许自动下载 MANIQA 权重

import pyiqa
from basicsr.utils import img2tensor

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)
    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

def dict2str(opt, indent=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += ' ' * (indent * 2) + "]\n"
        else:
            msg += ' ' * (indent * 2) + f"{k}: {v}\n"
    return msg

def main():
    parser = argparse.ArgumentParser(description="Image Quality Assessment Script")
    parser.add_argument("--inp_imgs", nargs="+", required=True)
    parser.add_argument("--gt_imgs",  nargs="+", required=True)
    parser.add_argument("--log",      type=str,  required=True)
    parser.add_argument("--log_name", type=str,  default='METRICS')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(CUSTOM_TORCH_HOME, exist_ok=True)
    os.makedirs(args.log, exist_ok=True)

    try:
        args.log_name = args.inp_imgs[0].split('/')[8]
    except IndexError:
        args.log_name = 'METRICS'
    setup_logger('base', args.log, f'test_{args.log_name}', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info("===== Configuration =====")
    logger.info(dict2str(vars(args)))
    logger.info(f"TORCH_HOME set to: {os.environ.get('TORCH_HOME')}")
    logger.info("==========================\n")

    logger.info("Initializing IQA metrics...")
    iqa_metrics = {
        'PSNR':    pyiqa.create_metric('psnr',    test_y_channel=True, color_space='ycbcr').to(device),
        'SSIM':    pyiqa.create_metric('ssim',    test_y_channel=True, color_space='ycbcr').to(device),
        'LPIPS':   pyiqa.create_metric('lpips',   device=device),
        'DISTS':   pyiqa.create_metric('dists',   device=device),
        'CLIPIQA': pyiqa.create_metric('clipiqa', device=device),
        'NIQE':    pyiqa.create_metric('niqe',    device=device),
        'MUSIQ':   pyiqa.create_metric('musiq',   device=device),
        # ★ 取消注释，允许自动下载 ViT 权重
        'MANIQA':  pyiqa.create_metric('maniqa-pipal', device=device),
    }
    fid_metric = pyiqa.create_metric('fid', device=device)
    logger.info("IQA metrics initialized.\n")

    if len(args.inp_imgs) != len(args.gt_imgs):
        logger.error("inp_imgs and gt_imgs must have the same number of directories.")
        sys.exit(1)

    init_imgs_names = []
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx]
        img_gt_list = sorted(glob.glob(os.path.join(gt_dir,   '*.png')))
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))
        dir_name = os.path.basename(os.path.normpath(init_dir))
        init_imgs_names.append(dir_name)
        logger.info(f"Directory [{dir_name}]: {len(img_gt_list)} GT vs {len(img_sr_list)} SR images.")
        assert len(img_gt_list) == len(img_sr_list), f"Mismatch: {dir_name}"

    logger.info("\n===== Starting Evaluation =====\n")

    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx]
        img_gt_list = sorted(glob.glob(os.path.join(gt_dir,   '*.png')))
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))
        dir_name = init_imgs_names[dir_idx]

        metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}
        logger.info(f"Testing Directory: [{dir_name}]")

        for img_idx, sr_path in enumerate(img_sr_list):
            gt_path  = img_gt_list[img_idx]
            img_name = os.path.basename(sr_path)
            start_time = time.time()

            sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            if sr_img is None or gt_img is None:
                logger.warning(f"Image read failed for {img_name}. Skipping.")
                continue

            sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
            gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

            with torch.no_grad():
                metrics = {}
                for name, metric in iqa_metrics.items():
                    # ★ 加入 MANIQA 到无参考指标列表
                    if name in ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA']:
                        metrics[name] = metric(sr_tensor).item()
                    else:
                        metrics[name] = metric(sr_tensor, gt_tensor).item()

            for name in metrics_accum:
                metrics_accum[name] += metrics[name]

            runtime = time.time() - start_time
            metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            logger.info(f"{dir_name}/{img_name} | {metrics_str} | Runtime: {runtime:.2f} sec")

        num_images  = len(img_sr_list)
        avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}

        fid_start = time.time()
        fid_value = fid_metric(gt_dir, init_dir).item()
        fid_runtime = time.time() - fid_start

        avg_metrics_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger.info(f"\n===== Average Metrics for [{dir_name}] =====\n{avg_metrics_str} | FID: {fid_value:.6f} | FID Runtime: {fid_runtime:.2f} sec\n")

    logger.info("===== Evaluation Completed =====")
    logger.info(f"All used weights are saved to: {CUSTOM_TORCH_HOME}")

if __name__ == "__main__":
    main()