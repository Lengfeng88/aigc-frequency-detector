# 在 Python 中临时运行：
from utils.dataset import AIGCFrequencyDataset
ds = AIGCFrequencyDataset('data')
train_s, val_s = ds.split_dataset(0.8, seed=42)
labels = [lbl for _, lbl in val_s]
print("验证集标签:", labels[:10])  # 打印前10个
print("真实图数量:", sum(1 for l in labels if l == 0))
print("AI图数量:", sum(1 for l in labels if l == 1))