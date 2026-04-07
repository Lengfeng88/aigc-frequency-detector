import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils.dataset import AIGCFrequencyDataset
from utils.freq_residual import preprocess_for_fra  # ← 关键：使用 FRA
from models.mobilenetv3_freq import create_freq_model, train_one_epoch, validate, save_model


class SubsetDataset(Dataset):
    def __init__(self, samples, target_size=224):
        self.samples = samples
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        tensor = preprocess_for_fra(img_path, self.target_size)  # ← 单通道 (1, H, W)
        return tensor, label


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    SAVE_PATH = 'checkpoints/best_model.pth'
    TARGET_SIZE = 224

    os.makedirs('checkpoints', exist_ok=True)

    # 加载数据
    full_dataset = AIGCFrequencyDataset(data_root='data', target_size=TARGET_SIZE)
    train_samples, val_samples = full_dataset.split_dataset(train_ratio=0.8, seed=42)

    # 创建子集
    train_dataset = SubsetDataset(train_samples, target_size=TARGET_SIZE)
    val_dataset = SubsetDataset(val_samples, target_size=TARGET_SIZE)

    # 打印验证集分布
    val_labels = [label for _, label in val_dataset.samples]
    real_val = sum(1 for l in val_labels if l == 0)
    fake_val = sum(1 for l in val_labels if l == 1)
    print(f"🔍 验证集分布 → 真实: {real_val}, AI生成: {fake_val}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"📊 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # 模型
    model = create_freq_model(model_name='mobilenetv3', device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, SAVE_PATH)

    print(f"🎉 训练完成！最佳验证准确率: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()