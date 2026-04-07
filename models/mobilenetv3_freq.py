import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small

class FreqMobileNetV3(nn.Module):
    """
    频域残差检测器：输入为单通道高频残差图（1xHxW）
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # 加载 MobileNetV3 Small，但不加载预训练权重
        base = mobilenet_v3_small(weights=None)
        
        # 修改第一层卷积：输入通道从 3 → 1
        base.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        
        self.backbone = base
        # 替换分类头
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ========================
# 训练/验证函数（保持不变）
# ========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc


def create_freq_model(model_name='mobilenetv3', num_classes=2, device='cpu'):
    if model_name == 'mobilenetv3':
        model = FreqMobileNetV3(num_classes=num_classes)
    else:
        raise ValueError("当前仅支持 'mobilenetv3'（单通道输入）")
    
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ 模型 '{model_name}' 已创建，参数量: {param_count:.2f}M")
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"💾 模型已保存至: {path}")


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"📂 模型已从 {path} 加载")
    return model