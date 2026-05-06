import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils.dataset import AIGCFrequencyDataset
from utils.freq_residual import preprocess_for_fra  # Key point: Use FRA
from models.mobilenetv3_freq import create_freq_model, train_one_epoch, validate, save_model


class SubsetDataset(Dataset):
    def __init__(self, samples, target_size=224):
        self.samples = samples
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        tensor = preprocess_for_fra(img_path, self.target_size)  # single channel (1, H, W)
        return tensor, label


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    SAVE_PATH = 'checkpoints/best_model.pth'
    TARGET_SIZE = 224

    os.makedirs('checkpoints', exist_ok=True)

    # Loading data
    full_dataset = AIGCFrequencyDataset(data_root='data', target_size=TARGET_SIZE)
    train_samples, val_samples = full_dataset.split_dataset(train_ratio=0.8, seed=42)

    # Create a subset
    train_dataset = SubsetDataset(train_samples, target_size=TARGET_SIZE)
    val_dataset = SubsetDataset(val_samples, target_size=TARGET_SIZE)

    # Print the distribution of the validation set
    val_labels = [label for _, label in val_dataset.samples]
    real_val = sum(1 for l in val_labels if l == 0)
    fake_val = sum(1 for l in val_labels if l == 1)
    print(f"Validation set distribution → Real: {real_val}, AI-generated: {fake_val}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"training set: {len(train_dataset)} | Validation set: {len(val_dataset)}")

    # model
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

    print(f"Training complete! Optimal validation accuracy achieved: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
