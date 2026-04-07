# models/mobilevit_aigc.py
import torch
import torch.nn as nn
from timm import create_model

class MobileViT_AIGC(nn.Module):
    """
    基于 MobileViT-S 的 AIGC 检测器
    - 输入: (3, H, W) RGB 图像
    - 输出: 2 类 (0=真实, 1=AI生成)
    """
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        # 加载 MobileViT-S（ImageNet 预训练可选）
        self.backbone = create_model(
            'mobilevit_s',
            pretrained=pretrained,      # 设为 False 避免分布偏移
            num_classes=num_classes,
            in_chans=3
        )
        # 如果不加载预训练，需初始化分类头
        if not pretrained:
            self._init_weights()

    def _init_weights(self):
        for m in self.backbone.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)