import cv2
import numpy as np
import torch
from scipy.ndimage import zoom


def rgb_to_fft_mag(image: np.ndarray) -> np.ndarray:
    """
    Convert an HxWx3 RGB image to a frequency domain logarithmic magnitude spectrum (preserving spatial structure).
    Args:
       image(np.ndarray): Shape (H, W, 3), dtype=uint8 or float32

   Returns:
       np.ndarray: Shape (H, W, 3), values ​​normalized to [0, 1], suitable for CNN input.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    
    fft_channels = []
    for i in range(3):  # R, G, B the channels separately do FFT
        f = np.fft.fft2(image[:, :, i])
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-8)  # Avoid division by zero
        fft_channels.append(magnitude)
    
    freq_img = np.stack(fft_channels, axis=-1)  # (H, W, 3)
    
    # Global normalization to [0, 1]
    min_val = freq_img.min()
    max_val = freq_img.max()
    if max_val - min_val > 1e-6:
        freq_img = (freq_img - min_val) / (max_val - min_val)
    else:
        freq_img = np.zeros_like(freq_img)
        
    return freq_img


def rgb_to_dct_blocks(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    The image is divided into blocks and subjected to DCT. The energy of the high-frequency sub-block (bottom right 4x4) is extracted as features.
    Args:
       image (np.ndarray): (H, W, 3) RGB image
       block_size (int): DCT block size (default 8x8)
    Returns:
       np.ndarray: (H_b, W_b, 3), each position represents the high-frequency energy of that block
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    
    h, w = image.shape[:2]
    # Clipping to a value divisible by block_size
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    img_cropped = image[:h_crop, :w_crop]
    
    dct_energy_list = []
    for c in range(3):
        channel = img_cropped[:, :, c]
        energy_map = []
        for i in range(0, h_crop, block_size):
            row_energies = []
            for j in range(0, w_crop, block_size):
                block = channel[i:i+block_size, j:j+block_size]
                dct_block = cv2.dct(block)  # shape: (8, 8)
                # Extract the high-frequency region: bottom right 4x4 (index [4:, 4:])
                high_freq = dct_block[4:, 4:]  # shape: (4, 4)
                energy = np.linalg.norm(high_freq)  # L2 Norm as energy
                row_energies.append(energy)
            energy_map.append(row_energies)
        dct_energy_list.append(np.array(energy_map))  # (H_b, W_b)
    
    # 合并为 (H_b, W_b, 3)
    dct_energy = np.stack(dct_energy_list, axis=-1)
    
    # Global normalization to [0, 1]
    min_val = dct_energy.min()
    max_val = dct_energy.max()
    if max_val - min_val > 1e-6:
        dct_energy = (dct_energy - min_val) / (max_val - min_val)
    else:
        dct_energy = np.zeros_like(dct_energy)
        
    return dct_energy


def preprocess_image_for_freq_cnn(
    image_path: str,
    target_size: int = 224,
    mode: str = 'fft'
) -> torch.Tensor:
    """
    Load the image from the file path and preprocess it into a frequency domain tensor.
    Args:
       image_path (str): Image path
       target_size (int): Size of the input network (e.g., 2^24)
       mode (str): 'fft' or 'dct'
    Returns:
       torch.Tensor: Shape (3, H, W), dtype=torch.float32
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    
    if mode == 'fft':
        freq_img = rgb_to_fft_mag(img_resized)
    elif mode == 'dct':
        dct_energy = rgb_to_dct_blocks(img_resized, block_size=8)
        # Upsampling target_size x target_size
        h_b, w_b, c = dct_energy.shape
        scale_h = target_size / h_b
        scale_w = target_size / w_b
        freq_img = zoom(dct_energy, (scale_h, scale_w, 1), order=1)  # Bilinear interpolation
    else:
        raise ValueError("mode must be 'fft' or 'dct'")
    
    # Convert to PyTorch tensors: (H, W, C) → (C, H, W)
    tensor = torch.from_numpy(freq_img).permute(2, 0, 1).float()
    return tensor


# ========================
# Visualization helper functions (supports FFT and DCT)
# ========================
def visualize_freq_spectrum(
    image_path: str,
    save_path: str = None,
    mode: str = 'fft'
):
    """
    Visualize the raw image and its frequency domain features (FFT amplitude spectrum or DCT high-frequency energy map).
    Args:
       image_path (str): Input image path
       save_path (str): Save path (displays a window if None)
       mode (str): 'fft' or 'dct'
    """
    import matplotlib.pyplot as plt
    
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    if mode == 'fft':
        freq_img = rgb_to_fft_mag(img_rgb)
        title = "FFT Magnitude (Log Scale)"
    elif mode == 'dct':
        dct_energy = rgb_to_dct_blocks(img_rgb, block_size=8)
        # Upsampling is used for visualization (preserving the original image size).
        h, w = img_rgb.shape[:2]
        h_b, w_b, _ = dct_energy.shape
        scale_h = h / h_b
        scale_w = w / w_b
        freq_img = zoom(dct_energy, (scale_h, scale_w, 1), order=1)
        title = "DCT High-Freq Energy"
    else:
        raise ValueError("mode must be 'fft' or 'dct'")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Use heatmaps to highlight energy differences
    axes[1].imshow(freq_img[:, :, 1], cmap='hot')  # Displaying green channel energy
    axes[1].set_title(title)
    axes[1].axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"The visualization has been saved to: {save_path}")
    else:
        plt.show()
