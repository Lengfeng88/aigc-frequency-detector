# utils/enhanced_preprocess.py
import torch
from torchvision import transforms
from PIL import Image
from gradient_variance_cuda import forward

def load_and_preprocess_image(image_path, target_size=256):
    """
    Load image and apply gradient-variance fusion preprocessing
    Returns: [3, H, W] tensor on GPU
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)  # [3, H, W]
    
    # Move to GPU and add batch dimension
    img_tensor = img_tensor.unsqueeze(0).cuda()  # [1, 3, H, W]
    
    # Apply custom fusion feature
    fused_features = forward(img_tensor, alpha=0.6, beta=0.4)
    
    return fused_features.squeeze(0)  # [3, H, W]