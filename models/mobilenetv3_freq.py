import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small

class FreqMobileNetV3(nn.Module):
    """
    Frequency domain residual detector: The input is a single-channel high-frequency residual map (1xHxW).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Load MobileNetV3 Small, but do not load pre-trained weights.
        base = mobilenet_v3_small(weights=None)
        
        # Modify the first convolutional layer: input channels change from 3 to 1.
        base.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        
        self.backbone = base
        # Replace category header
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ========================
# Training/validation function (remains unchanged)
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
        raise ValueError("Currently only supports 'mobilenetv3'（Single-channel input）")
    
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model '{model_name}' Created, number of parameters: {param_count:.2f}M")
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"The model has been saved to: {path}")


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"The model has been loaded from {path}")
    return model
