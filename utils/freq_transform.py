import cv2
import numpy as np
import torch
from scipy.ndimage import zoom


def rgb_to_fft_mag(image: np.ndarray) -> np.ndarray:
    """
    将 HxWx3 RGB 图像转换为频域对数幅度谱（保留空间结构）。
    
    Args:
        image (np.ndarray): 形状为 (H, W, 3)，dtype=uint8 或 float32
        
    Returns:
        np.ndarray: 形状为 (H, W, 3)，值已归一化到 [0, 1]，适合 CNN 输入
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    
    fft_channels = []
    for i in range(3):  # R, G, B 通道分别做 FFT
        f = np.fft.fft2(image[:, :, i])
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-8)  # 避免除零
        fft_channels.append(magnitude)
    
    freq_img = np.stack(fft_channels, axis=-1)  # (H, W, 3)
    
    # 全局归一化到 [0, 1]
    min_val = freq_img.min()
    max_val = freq_img.max()
    if max_val - min_val > 1e-6:
        freq_img = (freq_img - min_val) / (max_val - min_val)
    else:
        freq_img = np.zeros_like(freq_img)
        
    return freq_img


def rgb_to_dct_blocks(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    将图像分块进行 DCT，提取高频子块（右下 4x4）的能量作为特征。
    
    Args:
        image (np.ndarray): (H, W, 3) RGB 图像
        block_size (int): DCT 分块大小（默认 8x8）
        
    Returns:
        np.ndarray: (H_b, W_b, 3)，每个位置表示该块的高频能量
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    
    h, w = image.shape[:2]
    # 裁剪到 block_size 整除
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
                # 提取高频区域：右下 4x4（索引 [4:, 4:]）
                high_freq = dct_block[4:, 4:]  # shape: (4, 4)
                energy = np.linalg.norm(high_freq)  # L2 范数作为能量
                row_energies.append(energy)
            energy_map.append(row_energies)
        dct_energy_list.append(np.array(energy_map))  # (H_b, W_b)
    
    # 合并为 (H_b, W_b, 3)
    dct_energy = np.stack(dct_energy_list, axis=-1)
    
    # 全局归一化到 [0, 1]
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
    从文件路径加载图像，预处理为频域张量。
    
    Args:
        image_path (str): 图像路径
        target_size (int): 输入网络的尺寸（如 224）
        mode (str): 'fft' 或 'dct'
        
    Returns:
        torch.Tensor: 形状为 (3, H, W)，dtype=torch.float32
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    
    if mode == 'fft':
        freq_img = rgb_to_fft_mag(img_resized)
    elif mode == 'dct':
        dct_energy = rgb_to_dct_blocks(img_resized, block_size=8)
        # 上采样回 target_size x target_size
        h_b, w_b, c = dct_energy.shape
        scale_h = target_size / h_b
        scale_w = target_size / w_b
        freq_img = zoom(dct_energy, (scale_h, scale_w, 1), order=1)  # 双线性插值
    else:
        raise ValueError("mode 必须是 'fft' 或 'dct'")
    
    # 转为 PyTorch 张量: (H, W, C) → (C, H, W)
    tensor = torch.from_numpy(freq_img).permute(2, 0, 1).float()
    return tensor


# ========================
# 可视化辅助函数（支持 FFT 和 DCT）
# ========================
def visualize_freq_spectrum(
    image_path: str,
    save_path: str = None,
    mode: str = 'fft'
):
    """
    可视化原始图像和其频域特征（FFT 幅度谱 或 DCT 高频能量图）。
    
    Args:
        image_path (str): 输入图像路径
        save_path (str): 保存路径（若为 None 则显示窗口）
        mode (str): 'fft' 或 'dct'
    """
    import matplotlib.pyplot as plt
    
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    if mode == 'fft':
        freq_img = rgb_to_fft_mag(img_rgb)
        title = "FFT Magnitude (Log Scale)"
    elif mode == 'dct':
        dct_energy = rgb_to_dct_blocks(img_rgb, block_size=8)
        # 上采样用于可视化（保持原图尺寸）
        h, w = img_rgb.shape[:2]
        h_b, w_b, _ = dct_energy.shape
        scale_h = h / h_b
        scale_w = w / w_b
        freq_img = zoom(dct_energy, (scale_h, scale_w, 1), order=1)
        title = "DCT High-Freq Energy"
    else:
        raise ValueError("mode 必须是 'fft' 或 'dct'")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 使用热力图突出能量差异
    axes[1].imshow(freq_img[:, :, 1], cmap='hot')  # 显示绿色通道能量
    axes[1].set_title(title)
    axes[1].axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ 可视化已保存至: {save_path}")
    else:
        plt.show()