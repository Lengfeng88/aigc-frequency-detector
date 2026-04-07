# profile_with_nsight.py
import torch
import torch.nn.functional as F

def original_gradient_variance(img, alpha=0.6, beta=0.4):
    B, C, H, W = img.shape
    
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)
    
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    local_mean = F.avg_pool2d(img, kernel_size=5, stride=1, padding=2)
    local_mean_sq = F.avg_pool2d(img**2, kernel_size=5, stride=1, padding=2)
    local_var = torch.clamp(local_mean_sq - local_mean**2, min=0)
    
    return alpha * grad_mag + beta * local_var

def main():
    # Ensure we're using GPU
    device = torch.device('cuda')
    img = torch.rand(1, 3, 256, 256, device=device)
    
    # Warmup
    for _ in range(5):
        _ = original_gradient_variance(img)
    
    # Main execution (this will be captured by Nsight)
    for i in range(10):  # Run multiple times for better visualization
        result = original_gradient_variance(img)
        torch.cuda.synchronize()  # Ensure all work is completed

if __name__ == "__main__":
    main()