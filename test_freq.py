# test_freq.py
import os
import cv2
import numpy as np
import torch
from utils.freq_transform import preprocess_image_for_freq_cnn, visualize_freq_spectrum

# Create test data directory
os.makedirs("data/fake", exist_ok=True)

# Generate a simulated AI-generated image with periodic texture (to trigger checkerboard artifacts).
H, W = 256, 256
img = np.zeros((H, W, 3), dtype=np.uint8)

# Add high-frequency periodic noise (to simulate a deconvolution checkerboard pattern).
for c in range(3):
    for i in range(H):
        for j in range(W):
            # Simple sine wave + random perturbation
            val = 128 + 60 * np.sin(2 * np.pi * i / 16) * np.cos(2 * np.pi * j / 16)
            img[i, j, c] = np.clip(val + np.random.randint(-10, 10), 0, 255)

cv2.imwrite("data/fake/synthetic_fake.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Test frequency domain preprocessing
fake_img = "data/fake/synthetic_fake.png"

try:
    tensor = preprocess_image_for_freq_cnn(fake_img, target_size=224)
    print("Frequency domain tensor shape:", tensor.shape)
    print("value range: {:.3f} to {:.3f}".format(tensor.min().item(), tensor.max().item()))
    
    # Visualize and save
    visualize_freq_spectrum(fake_img, save_path="freq_demo.png")
    print("Frequency domain visualization has been saved as freq_demo.png")
    
except Exception as e:
    print("erro:", e)
