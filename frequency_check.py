# debug_freq.py
from utils.freq_transform import visualize_freq_spectrum
import os

# Find a real image and an AI image.
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
    print("Generating frequency domain visualization...")
    visualize_freq_spectrum("data/fake/sd_001.png", "fake_dct.png", mode='dct')
    visualize_freq_spectrum("data/real/nature.jpg", "real_dct.png", mode='dct')
    print("The visualization has been saved as: debug_real_freq.png, debug_fake_freq.png")
else:
    print("Image not found")
