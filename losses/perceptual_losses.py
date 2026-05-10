import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['TORCH_HOME'] = '/root/.cache/torch'


class PDINOLoss(nn.Module):
    """
    P-DINO 感知损失（来自 PixelGen，修正为余弦距离版本）

    论文依据：
    - PixelGen (Ma et al. 2026) 使用冻结 DINOv2 最后一层 patch feature 的余弦距离
    - 最后一层（第12层）捕捉全局语义，浅层只有低级外观特征
    - 公式：L = 1 - mean_over_patches( cos_sim(f(ŷ)_p, f(y)_p) )
    - 数值范围 [0, 2]，语义相近图像对通常收敛到 0.02～0.15

    原始实现用 MSE，但 DINOv2 feature 的模长无语义含义，
    余弦距离只对齐方向（语义），更符合论文意图且数值更稳定。
    """

    def __init__(self, model_name: str = 'dinov2_vitb14'):
        super().__init__()

        # hub 已下载缓存，离线也能用
        self.dino = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True,
            force_reload=False,
        )
        self.dino.eval()
        self.dino.requires_grad_(False)

        # ImageNet 归一化统计量
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：[-1, 1] 范围的图像
        输出：DINOv2 归一化后，H/W 对齐到 patch_size=14 的倍数
        """
        x = (x + 1.0) / 2.0                    # [-1, 1] → [0, 1]
        x = (x - self.mean) / self.std          # ImageNet 归一化

        B, C, H, W = x.shape
        H_new = (H // 14) * 14                  # patch_size = 14
        W_new = (W // 14) * 14
        if H_new != H or W_new != W:
            x = F.interpolate(
                x, size=(H_new, W_new),
                mode='bilinear', align_corners=False,
            )
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, 3, H, W]，范围 [-1, 1]

        返回标量损失：
            L = 1 - mean_{b,p}( cos_sim(f(pred)_{b,p}, f(target)_{b,p}) )

        梯度只通过 pred 传播，target 用 no_grad 提取特征。
        """
        pred_input   = self.preprocess(pred.float())
        target_input = self.preprocess(target.float())

        # target 特征：冻结，不需要梯度
        with torch.no_grad():
            target_feats = self.dino.get_intermediate_layers(
                target_input, n=1
            )[0]                                # [B, num_patches, 768]

        # pred 特征：需要梯度传回生成模型
        pred_feats = self.dino.get_intermediate_layers(
            pred_input, n=1
        )[0]                                    # [B, num_patches, 768]

        # L2 normalize → 余弦相似度 → 1 - similarity = 余弦距离
        pred_norm   = F.normalize(pred_feats,          dim=-1)   # [B, P, 768]
        target_norm = F.normalize(target_feats.detach(), dim=-1)  # [B, P, 768]

        # 每个 patch 的余弦相似度，再对 batch 和 patch 求均值
        cos_sim = (pred_norm * target_norm).sum(dim=-1)           # [B, P]
        loss    = 1.0 - cos_sim.mean()

        return loss