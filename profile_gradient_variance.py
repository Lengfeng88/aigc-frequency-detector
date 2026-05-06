# profile_gradient_variance.py
import torch
import torch.nn.functional as F
import time
from torch.profiler import profile, record_function, ProfilerActivity

def original_gradient_variance(img, alpha=0.6, beta=0.4):
    """
    Native PyTorch Implementation: Local Gradient-Variance Fusion Feature
    Input: [B, C, H, W] RGB image (0-1 normalized)
    Output: [B, C, H, W] fused features
    """
    B, C, H, W = img.shape
    
    # Sobel operator (3x3)
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
    
    # Expanded to multi-channel (groups=C)
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)
    
    # Calculate gradient
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # Avoid division by zero
    
    # Local variance (5x5 window)
    # First calculate the local mean
    local_mean = F.avg_pool2d(img, kernel_size=5, stride=1, padding=2)
    # Calculate the local mean square
    local_mean_sq = F.avg_pool2d(img**2, kernel_size=5, stride=1, padding=2)
    # variance = E[X^2] - (E[X])^2
    local_var = torch.clamp(local_mean_sq - local_mean**2, min=0)
    
    # Fusion
    fused = alpha * grad_mag + beta * local_var
    return fused

def benchmark_latency():
    """Measure average delay"""
    print("=== Delay benchmark ===")
    img = torch.rand(1, 3, 256, 256).cuda()
    
    # Warmup
    for _ in range(10):
        _ = original_gradient_variance(img)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        out = original_gradient_variance(img)
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / iterations * 1000  # ms
    
    print(f"Input dimensions: {img.shape}")
    print(f"Average delay: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.2f} FPS\n")

def profile_operations():
    """Detailed performance analysis"""
    print("=== Detailed performance analysis (PyTorch Profiler) ===")
    img = torch.rand(1, 3, 256, 256).cuda()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with record_function("gradient_variance"):
            out = original_gradient_variance(img)
    
    # Printing Key Indicators
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        max_name_column_width=50,
        row_limit=10
    ))
    
    # Save the trace file (for Chrome://tracing).
    prof.export_chrome_trace("gradient_variance_trace.json")
    print("\n trace file saved: gradient_variance_trace.json")
    print("Open chrome://tracing in Chrome to view the detailed timeline.")

if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA device to run.")
    
    print("Spatial Domain Local Inconsistency Detection - Performance Analysis")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
    
    benchmark_latency()
    profile_operations()
