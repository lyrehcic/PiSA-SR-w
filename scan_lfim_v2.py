"""
scan_lfim_v2.py 终极修复版
"""
import os
import subprocess
import argparse
import itertools
import re

SCAN_CONFIG = {
    "mf_beta": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "hf_beta": [0.1],
    "lf_alpha": [0.0],
    "entropy_weight": [0.5],
    "cutoff_low": [0.1],
    "cutoff_mid": [0.3],
}

def run_cmd(cmd):
    print("\n▶ 运行:", cmd)
    return subprocess.run(cmd, shell=True).returncode

def parse_metrics_log(log_path):
    metrics = {}
    if not os.path.isfile(log_path):
        return metrics
    with open(log_path, 'r') as f:
        txt = f.read()

    metrics["PSNR"] = float(re.search(r"PSNR:\s*([\d.]+)", txt).group(1))
    metrics["SSIM"] = float(re.search(r"SSIM:\s*([\d.]+)", txt).group(1))
    metrics["LPIPS"] = float(re.search(r"LPIPS:\s*([\d.]+)", txt).group(1))
    metrics["DISTS"] = float(re.search(r"DISTS:\s*([\d.]+)", txt).group(1))
    metrics["CLIPIQA"] = float(re.search(r"CLIPIQA:\s*([\d.]+)", txt).group(1))
    metrics["NIQE"] = float(re.search(r"NIQE:\s*([\d.]+)", txt).group(1))
    metrics["MUSIQ"] = float(re.search(r"MUSIQ:\s*([\d.]+)", txt).group(1))
    metrics["MANIQA"] = float(re.search(r"MANIQA:\s*([\d.]+)", txt).group(1))
    metrics["FID"] = float(re.search(r"FID:\s*([\d.]+)", txt).group(1))
    return metrics

def main(args):
    os.makedirs(args.out, exist_ok=True)
    configs = [dict(zip(SCAN_CONFIG.keys(), c)) for c in itertools.product(*SCAN_CONFIG.values())]
    results = []

    for cfg in configs:
        tag = f"mf{cfg['mf_beta']}_hf{cfg['hf_beta']}"
        out_dir = os.path.join(args.out, tag)
        log_file = os.path.join(out_dir, "metrics.log")  # <== 这里改成 log，绝对不会变文件夹

        if os.path.exists(log_file) and not args.force:
            print(f"✅ 已存在 {tag}")
            res = parse_metrics_log(log_file)
            results.append({"tag": tag, "cfg": cfg, **res})
            continue

        run_cmd(f"""
        python test_pisasr_wavelora_lfim_v2.py \
        --pretrained_model_path {args.sd} \
        --pretrained_path {args.ckpt} \
        --process_size 512 --upscale 4 --default \
        --input_image {args.lr} --output_dir {out_dir} \
        --mf_beta {cfg['mf_beta']} --hf_beta {cfg['hf_beta']} \
        --lf_alpha {cfg['lf_alpha']} --entropy_weight {cfg['entropy_weight']} \
        --cutoff_low {cfg['cutoff_low']} --cutoff_mid {cfg['cutoff_mid']} \
        --align_method {args.align_method} --mixed_precision {args.mixed_precision}
        """)

        run_cmd(f"""
        python test_metrics.py \
        --inp_imgs {out_dir} \
        --gt_imgs {args.hr} \
        --log {log_file}
        """)

        res = parse_metrics_log(log_file)
        results.append({"tag": tag, "cfg": cfg, **res})

    # 输出 CSV
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("tag,mf_beta,hf_beta,PSNR,SSIM,LPIPS,DISTS,CLIPIQA,NIQE,MUSIQ,MANIQA,FID\n")
        for r in results:
            c = r["cfg"]
            f.write(f"{r['tag']},{c['mf_beta']},{c['hf_beta']},{r['PSNR']},{r['SSIM']},{r['LPIPS']},{r['DISTS']},{r['CLIPIQA']},{r['NIQE']},{r['MUSIQ']},{r['MANIQA']},{r['FID']}\n")

    print("\n✅ CSV 已生成：", csv_path)
    print("\n🏆 最佳结果（按 MANIQA）：")
    best = max(results, key=lambda x: x["MANIQA"])
    print(best["tag"], best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sd", type=str, required=True)
    parser.add_argument("--lr", type=str, required=True)
    parser.add_argument("--hr", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--align_method", type=str, default="adain")
    parser.add_argument("--mixed_precision", type=str, default="fp32")
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())