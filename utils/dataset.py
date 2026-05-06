import os
import random
from pathlib import Path
from torch.utils.data import Dataset
from utils.freq_transform import preprocess_image_for_freq_cnn

class AIGCFrequencyDataset(Dataset):
    """
    A binary classification dataset of AI-generated images vs. real images (frequency domain input).
    Directory structure requirements:
    data/
    ├── real/      # Real image (label=0)
    └── fake/      # AI-generated image (label=1)
    """
    def __init__(self, data_root: str, target_size: int = 224, transform=None):
        """
        Args:
            data_root (str): Data root directory path (e.g., 'data')
            target_size (int): Image size of input network
            transform: Reserved interface (current frequency domain transformation is already built-in, usually set to None)
        """
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.transform = transform
        
        # Supported image extensions
        IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Load the actual image path (label 0)
        real_dir = self.data_root / 'real'
        self.real_paths = [
            str(p) for p in real_dir.rglob('*') 
            if p.suffix.lower() in IMG_EXTENSIONS
        ]
        
        # Load the path to the AI-generated image (Tag 1)
        fake_dir = self.data_root / 'fake'
        self.fake_paths = [
            str(p) for p in fake_dir.rglob('*') 
            if p.suffix.lower() in IMG_EXTENSIONS
        ]
        
        # Merge paths and tags
        self.samples = []
        self.samples.extend([(path, 0) for path in self.real_paths])
        self.samples.extend([(path, 1) for path in self.fake_paths])
        
        print(f"Loading complete: {len(self.real_paths)} Real images, {len(self.fake_paths)} AI-generated images")
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_root}! Please check the directory structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Core: Converting images into frequency domain tensors
            freq_tensor = preprocess_image_for_freq_cnn(img_path, self.target_size)
            return freq_tensor, label
        except Exception as e:
            raise RuntimeError(f"Image processing failed {img_path}: {e}")

    def split_dataset(self, train_ratio=0.8, seed=42):
        """
        Split the dataset into training and validation sets (maintaining class balance).
        Return two lists of subsamples: (train_samples, val_samples)
        """
        random.seed(seed)
        
        # Disrupt real and fake samples respectively
        real_samples = [(p, 0) for p in self.real_paths]
        fake_samples = [(p, 1) for p in self.fake_paths]
        
        random.shuffle(real_samples)
        random.shuffle(fake_samples)
        
        # Divided by proportion
        n_real_train = int(len(real_samples) * train_ratio)
        n_fake_train = int(len(fake_samples) * train_ratio)
        
        train_samples = real_samples[:n_real_train] + fake_samples[:n_fake_train]
        val_samples = real_samples[n_real_train:] + fake_samples[n_fake_train:]
        
        # Shuffle the merged list again
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        
        return train_samples, val_samples
