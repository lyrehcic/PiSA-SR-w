import os
import re
import glob

# ===================== 配置路径（直接用你的路径）=====================
BASE_DIR = "/data/wyb/PiSA-SR/preset/datasets/benchmark_realsr/DE_dataset"
OUTPUT_FILE = os.path.join(BASE_DIR, "all_metrics_hl_DE.txt")

# 存储结果
results = []

# 1. 遍历所有 results_hl_DE_数字 文件夹
for folder_name in os.listdir(BASE_DIR):
    if not folder_name.startswith("results_hl_DE_"):
        continue

    # 提取数字 例如 9000
    match = re.search(r"results_hl_DE_(\d+)", folder_name)
    if not match:
        continue
    num = int(match.group(1))
    folder_path = os.path.join(BASE_DIR, folder_name)

    # 2. 找 metrics 目录下的 .log 文件
    log_dir = os.path.join(folder_path, "metrics")
    if not os.path.exists(log_dir):
        continue

    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        continue

    # 取第一个 log（通常只有一个）
    log_file = log_files[0]

    # 3. 读取指标行
    target_line = None
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if "===== Average Metrics for" in line:
                target_line = line.strip()
            elif target_line is not None and "PSNR:" in line:
                # 保存：数字、文件夹名、指标行
                results.append((num, folder_name, line.strip()))
                break

# 4. 按数字从小到大排序
results.sort(key=lambda x: x[0])

# 5. 写入总文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for num, fname, metric_line in results:
        f.write(f"===== Average Metrics for [{fname}] =====\n")
        f.write(metric_line + "\n\n")

print(f"✅ 整理完成！所有指标已保存到：\n{OUTPUT_FILE}")
print(f"📊 一共整理了 {len(results)} 个模型结果")