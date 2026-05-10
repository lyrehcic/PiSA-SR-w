import torch
from pisasr_wave_hl import PiSASR
from types import SimpleNamespace

# 👇 这里填你的模型路径（你现在用的 18501.pkl 也可以）
CKPT_PATH = "/data/wyb/PiSA-SR/experiments/dataset-LSDIR+FFHQ/train-pisasr-wavelora-hl-lr1_linear_272M/checkpoints/model_18501.pkl"

# 构造最小配置
args = SimpleNamespace(
    pretrained_model_path="/data/checkpoints/OSEDiff/ckpt/stable-diffusion-2-1-base",
    resume_ckpt=CKPT_PATH,
    wave_dim=None,
    wave_res=32,
    mlp_ratio=1.0,
    wave_scale=1.0,
    lora_rank_unet_pix=4,
    timesteps1=1,
)

# 加载你现在的模型
model = PiSASR(args)
model.eval()

# 统计所有可训练/总参数
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e9, trainable / 1e9

total_B, trainable_B = count_params(model)
print("\n" + "="*60)
print(f"✅ 你当前模型：PiSASR_wave_hl + Dual LoRA + wave_scale=1.0")
print(f"📏 总参数量：{total_B:.3f} B")
print(f"🔥 可训练参数量：{trainable_B:.3f} B")
print("="*60)