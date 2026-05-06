# Temporarily run in Python:
from utils.dataset import AIGCFrequencyDataset
ds = AIGCFrequencyDataset('data')
train_s, val_s = ds.split_dataset(0.8, seed=42)
labels = [lbl for _, lbl in val_s]
print("Validation set tags:", labels[:10])  # Print the first 10
print("Number of real images:", sum(1 for l in labels if l == 0))
print("Number of AI graphs:", sum(1 for l in labels if l == 1))
