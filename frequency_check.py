# debug_freq.py
from utils.freq_transform import visualize_freq_spectrum
import os

# 找一张真实图和一张 AI 图
real_img = None
fake_img = None

for root, _, files in os.walk("data/real"):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            real_img = os.path.join(root, f)
            break
    if real_img: break

for root, _, files in os.walk("data/fake"):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            fake_img = os.path.join(root, f)
            break
    if fake_img: break

if real_img and fake_img:
    print("🔍 正在生成频域可视化...")
    visualize_freq_spectrum("data/fake/sd_001.png", "fake_dct.png", mode='dct')
    visualize_freq_spectrum("data/real/nature.jpg", "real_dct.png", mode='dct')
    print("✅ 可视化已保存为: debug_real_freq.png, debug_fake_freq.png")
else:
    print("❌ 未找到图像")