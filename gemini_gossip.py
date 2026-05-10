import os
import shutil

# 路径配置
root_dir = "/data/wyb/PiSA-SR/Urban100"
src_dir = os.path.join(root_dir, "image_SRF_4")
lr_out = os.path.join(root_dir, "LR")
hr_out = os.path.join(root_dir, "HR")

# 创建输出文件夹
os.makedirs(lr_out, exist_ok=True)
os.makedirs(hr_out, exist_ok=True)

# 收集并按序号排序
lr_list = []
hr_list = []

for fname in os.listdir(src_dir):
    if fname.endswith(".png"):
        if "_LR.png" in fname:
            lr_list.append(fname)
        elif "_HR.png" in fname:
            hr_list.append(fname)

# 按图片编号排序（保证 001、002...顺序）
def sort_key(name):
    # 提取 img_001 中的数字
    return int(name.split("img_")[-1].split("_")[0])

lr_list.sort(key=sort_key)
hr_list.sort(key=sort_key)

# 复制文件
for fname in lr_list:
    shutil.copy2(
        os.path.join(src_dir, fname),
        os.path.join(lr_out, fname)
    )

for fname in hr_list:
    shutil.copy2(
        os.path.join(src_dir, fname),
        os.path.join(hr_out, fname)
    )

print(f"完成！")
print(f"LR 数量：{len(lr_list)}")
print(f"HR 数量：{len(hr_list)}")
print(f"LR保存目录：{lr_out}")
print(f"HR保存目录：{hr_out}")