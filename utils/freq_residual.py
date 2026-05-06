# utils/freq_residual.py
import cv2
import numpy as np
import torch
from scipy.ndimage import zoom

def compute_phase_congruency(image: np.ndarray, sigma=2.0) -> np.ndarray:
    """
    Calculate the approximate phase congruency of the image.
    Simulate multi-scale edge responses using a Gabor filter bank
    Returns: (H, W) phase congruency map (values ​​∈ [0,1])
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Convert to grayscale (RGB → Y)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale Gabor filtering (simplified version: 2-scale)
    scales = [3, 5]
    pc_sum = np.zeros_like(gray)
    
    for scale in scales:
        # Generate Gabor cores (orientation 0°, 90°).
        ksize = 2 * scale + 1
        gabor0 = cv2.getGaborKernel((ksize, ksize), sigma, 0, scale, 1, 0, ktype=cv2.CV_32F)
        gabor90 = cv2.getGaborKernel((ksize, ksize), sigma, np.pi/2, scale, 1, 0, ktype=cv2.CV_32F)
        
        # Filtering
        filtered0 = cv2.filter2D(gray, cv2.CV_32F, gabor0)
        filtered90 = cv2.filter2D(gray, cv2.CV_32F, gabor90)
        
        # Amplitude & Phase Approximation
        mag = np.sqrt(filtered0**2 + filtered90**2)
        # Phase consistency ≈ mag / (sum |filter| + ε)
        eps = 1e-6
        pc = mag / (np.abs(filtered0) + np.abs(filtered90) + eps)
        pc_sum += pc
    
    # Normalization to [0,1]
    pc_sum = np.clip(pc_sum, 0, 1)
    return pc_sum


def extract_frequency_residual(image: np.ndarray, target_size=224) -> np.ndarray:
    """
    Extracting frequency domain residual features:
       1. Calculate the phase consistency map (PC)
       2. Perform DCT on the PC map and extract the high-frequency block energy (as the "residual")
       3. Output: (H_b, W_b, 1) → Single-channel high-frequency residual map
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    
    # Step 1: Phase Consistency Map
    pc_map = compute_phase_congruency(image)
    
    # Step 2: Perform DCT segmentation (8x8) on the PC image and extract high-frequency energy.
    h, w = pc_map.shape
    block_size = 8
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    pc_cropped = pc_map[:h_crop, :w_crop]
    
    energy_map = []
    for i in range(0, h_crop, block_size):
        row_energies = []
        for j in range(0, w_crop, block_size):
            block = pc_cropped[i:i+block_size, j:j+block_size]
            dct_block = cv2.dct(block)
            high_freq = dct_block[4:, 4:]  # 4x4 high frequency
            energy = np.linalg.norm(high_freq)
            row_energies.append(energy)
        energy_map.append(row_energies)
    
    energy_map = np.array(energy_map)  # (H_b, W_b)
    
    # Normalization
    min_val = energy_map.min()
    max_val = energy_map.max()
    if max_val - min_val > 1e-6:
        energy_map = (energy_map - min_val) / (max_val - min_val)
    else:
        energy_map = np.zeros_like(energy_map)
    
    # Upsampling target_size
    h_b, w_b = energy_map.shape
    scale_h, scale_w = target_size / h_b, target_size / w_b
    residual = zoom(energy_map, (scale_h, scale_w), order=1)  # (H, W)
    
    # Return to single channel (H, W, 1)
    return residual[:, :, np.newaxis]


def preprocess_for_fra(image_path: str, target_size=224) -> torch.Tensor:
    """
    Frequency domain residual input: Single-channel high-frequency residual plot (more robust than DCT)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    
    residual = extract_frequency_residual(img_resized, target_size)
    # residual shape: (H, W, 1) → 转为 (1, H, W)
    tensor = torch.from_numpy(residual).permute(2, 0, 1).float()  # (1, H, W)
    return tensor
