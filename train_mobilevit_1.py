# train_mobilevit.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.mobilevit_aigc import MobileViT_AIGC

# ----------------------------
# 数据集类（直接加载原始图像）
# ----------------------------
class AIGCRGBDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),  # MobileViT 推荐 256x256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ----------------------------
# 辅助函数：划分数据集
# ----------------------------
def load_and_split_data(data_root, train_ratio=0.8, seed=42):
    data_root = Path(data_root)
    real_paths = list(data_root.glob("real/*.*"))
    fake_paths = list(data_root.glob("fake/*.*"))
    
    import random
    random.seed(seed)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    n_real_train = int(len(real_paths) * train_ratio)
    n_fake_train = int(len(fake_paths) * train_ratio)
    
    train_samples = [(str(p), 0) for p in real_paths[:n_real_train]] + \
                    [(str(p), 1) for p in fake_paths[:n_fake_train]]
    val_samples = [(str(p), 0) for p in real_paths[n_real_train:]] + \
                  [(str(p), 1) for p in fake_paths[n_fake_train:]]
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    print(f"✅ 加载完成: {len(real_paths)} 真实图, {len(fake_paths)} AI图")
    print(f"📊 训练集: {len(train_samples)}, 验证集: {len(val_samples)}")
    return train_samples, val_samples


# ----------------------------
# 训练/验证循环
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


# ----------------------------
# 主函数
# ----------------------------
def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16  # MobileViT 较大，WSL2 CPU 建议 ≤16
    EPOCHS = 20
    LR = 1e-4
    SAVE_PATH = 'checkpoints/best_mobilevit.pth'
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # 加载数据
    train_samples, val_samples = load_and_split_data('data')
    
    # 创建数据集
    train_dataset = AIGCRGBDataset(train_samples)
    val_dataset = AIGCRGBDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 模型（不加载 ImageNet 预训练，避免分布偏移）
    model = MobileViT_AIGC(num_classes=2, pretrained=False).to(DEVICE)
    print(f"✅ MobileViT-S 已创建，参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 模型已保存至: {SAVE_PATH}")
    
    print(f"🎉 训练完成！最佳验证准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()