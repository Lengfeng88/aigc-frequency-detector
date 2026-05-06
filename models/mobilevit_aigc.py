# models/mobilevit_aigc.py
import torch
import torch.nn as nn
from timm import create_model

class MobileViT_AIGC(nn.Module):
    """
    AIGC Detector Based on MobileViT-S
       - Input: (3, H, W) RGB image
       - Output: 2 classes (0=real, 1=AI generated)
    """
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        # Load MobileViT-S (ImageNet pre-training optional)
        self.backbone = create_model(
            'mobilevit_s',
            pretrained=pretrained,      # Set to False to avoid distribution shift.
            num_classes=num_classes,
            in_chans=3
        )
        # If pre-training is not loaded, the classification head needs to be initialized.
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
