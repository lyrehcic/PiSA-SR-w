import subprocess
import os

# 生成所有需要运行的步数：20501 到 28001，间隔 500
steps = list(range(15501, 22001 + 1, 500))

# 循环执行每一个模型
for step in steps:
    print(f"=" * 60)
    print(f"🚀 正在处理模型步数：{step}")
    print(f"=" * 60)

    # ==================== 1. 运行超分推理 ====================
    cmd1 = [
        "python", "test_pisasr_wavelora.py",
        "--pretrained_model_path", "/data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base",
        "--pretrained_path", f"/data/wyb/PiSA-SR/experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-lr1_linear_scale1/checkpoints/model_{step}.pkl",
        "--process_size", "512",
        "--upscale", "4",
        "--input_image", "/data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/test_LR",
        "--output_dir", f"/data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/scale1_wave_hl_AB/results_scale1_wave_hl_AB_{step}",
        "--default"
    ]

    print("🔹 执行超分推理...")
    subprocess.run(cmd1, check=True)  # 等待命令执行完成

    # ==================== 2. 运行指标计算 ====================
    cmd2 = [
        "python", "test_metrics.py",
        "--inp_imgs", f"/data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/scale1_wave_hl_AB/results_scale1_wave_hl_AB_{step}",
        "--gt_imgs", "/data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/test_HR",
        "--log", f"/data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/scale1_wave_hl_AB/results_scale1_wave_hl_AB_{step}/metrics"
    ]

    print("🔹 计算指标...")
    subprocess.run(cmd2, check=True)

    print(f"✅ 步骤 {step} 全部完成！\n")

print("🎉 所有模型测试 + 指标计算 全部执行完毕！")