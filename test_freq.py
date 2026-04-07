# test_freq.py
import os
import cv2
import numpy as np
import torch
from utils.freq_transform import preprocess_image_for_freq_cnn, visualize_freq_spectrum

# 创建测试数据目录
os.makedirs("data/fake", exist_ok=True)

# 生成一张带周期性纹理的模拟 AI 生成图（用于触发棋盘格伪影）
H, W = 256, 256
img = np.zeros((H, W, 3), dtype=np.uint8)

# 添加高频周期性噪声（模拟反卷积棋盘格）
for c in range(3):
    for i in range(H):
        for j in range(W):
            # 简单正弦波 + 随机扰动
            val = 128 + 60 * np.sin(2 * np.pi * i / 16) * np.cos(2 * np.pi * j / 16)
            img[i, j, c] = np.clip(val + np.random.randint(-10, 10), 0, 255)

cv2.imwrite("data/fake/synthetic_fake.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# 测试频域预处理
fake_img = "data/fake/synthetic_fake.png"

try:
    tensor = preprocess_image_for_freq_cnn(fake_img, target_size=224)
    print("✅ 频域张量形状:", tensor.shape)
    print("✅ 值范围: {:.3f} to {:.3f}".format(tensor.min().item(), tensor.max().item()))
    
    # 可视化并保存
    visualize_freq_spectrum(fake_img, save_path="freq_demo.png")
    print("✅ 频域可视化已保存为 freq_demo.png")
    
except Exception as e:
    print("❌ 错误:", e)