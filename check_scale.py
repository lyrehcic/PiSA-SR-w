import torch

# 加载你的模型
sd = torch.load("/data/wyb/PiSA-SR/experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-lr1_linear_272M/checkpoints/model_18501.pkl")

print("=== pix wave scales ===")
for k, v in sd["state_dict_pix_wave"].items():
    if "scale" in k:
        print(f"{k}: {v}")

print("\n=== sem wave scales ===")
for k, v in sd["state_dict_sem_wave"].items():
    if "scale" in k:
        print(f"{k}: {v}")