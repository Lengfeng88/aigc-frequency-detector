import os
import random
from pathlib import Path
from torch.utils.data import Dataset
from utils.freq_transform import preprocess_image_for_freq_cnn

class AIGCFrequencyDataset(Dataset):
    """
    AI 生成图像 vs 真实图像的二分类数据集（频域输入）。
    
    目录结构要求：
    data/
    ├── real/      # 真实图像（标签=0）
    └── fake/      # AI 生成图像（标签=1）
    """
    def __init__(self, data_root: str, target_size: int = 224, transform=None):
        """
        Args:
            data_root (str): 数据根目录路径（如 'data'）
            target_size (int): 输入网络的图像尺寸
            transform: 预留接口（当前频域变换已内置，通常设为 None）
        """
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.transform = transform
        
        # 支持的图像扩展名
        IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 加载真实图像路径（标签 0）
        real_dir = self.data_root / 'real'
        self.real_paths = [
            str(p) for p in real_dir.rglob('*') 
            if p.suffix.lower() in IMG_EXTENSIONS
        ]
        
        # 加载 AI 生成图像路径（标签 1）
        fake_dir = self.data_root / 'fake'
        self.fake_paths = [
            str(p) for p in fake_dir.rglob('*') 
            if p.suffix.lower() in IMG_EXTENSIONS
        ]
        
        # 合并路径与标签
        self.samples = []
        self.samples.extend([(path, 0) for path in self.real_paths])
        self.samples.extend([(path, 1) for path in self.fake_paths])
        
        print(f"✅ 加载完成: {len(self.real_paths)} 张真实图像, {len(self.fake_paths)} 张 AI 生成图像")
        if len(self.samples) == 0:
            raise ValueError(f"未在 {data_root} 中找到任何图像！请检查目录结构。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # 核心：将图像转为频域张量
            freq_tensor = preprocess_image_for_freq_cnn(img_path, self.target_size)
            return freq_tensor, label
        except Exception as e:
            raise RuntimeError(f"处理图像失败 {img_path}: {e}")

    def split_dataset(self, train_ratio=0.8, seed=42):
        """
        将数据集划分为训练集和验证集（保持类别平衡）
        返回两个子样本列表：(train_samples, val_samples)
        """
        random.seed(seed)
        
        # 分别打乱真实和伪造样本
        real_samples = [(p, 0) for p in self.real_paths]
        fake_samples = [(p, 1) for p in self.fake_paths]
        
        random.shuffle(real_samples)
        random.shuffle(fake_samples)
        
        # 按比例划分
        n_real_train = int(len(real_samples) * train_ratio)
        n_fake_train = int(len(fake_samples) * train_ratio)
        
        train_samples = real_samples[:n_real_train] + fake_samples[:n_fake_train]
        val_samples = real_samples[n_real_train:] + fake_samples[n_fake_train:]
        
        # 再次打乱合并后的列表
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        
        return train_samples, val_samples