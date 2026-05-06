# benchmark_speedup.py
import torch
import torch.nn.functional as F
import time
import statistics

# Import your custom operator
try:
    from gradient_variance_cuda import forward as custom_gradient_variance
    CUSTOM_OP_AVAILABLE = True
    print("Custom gradient-variance operator loaded")
except ImportError:
    CUSTOM_OP_AVAILABLE = False
    print("Custom operator not available")

def original_gradient_variance(img, alpha=0.6, beta=0.4):
    """Original PyTorch implementation"""
    B, C, H, W = img.shape
    
    # Sobel operators
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
    
    # Gradient calculation
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    # Local variance (5x5 window)
    local_mean = F.avg_pool2d(img, kernel_size=5, stride=1, padding=2)
    local_mean_sq = F.avg_pool2d(img**2, kernel_size=5, stride=1, padding=2)
    local_var = torch.clamp(local_mean_sq - local_mean**2, min=0)
    
    return alpha * grad_mag + beta * local_var

def benchmark_function(func, input_tensor, num_warmup=10, num_iterations=100):
    """Benchmark a function with CUDA events for accurate timing"""
    # Warmup
    for _ in range(num_warmup):
        _ = func(input_tensor)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = func(input_tensor)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)
    
    return statistics.mean(times), statistics.stdev(times)

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test different batch sizes and resolutions
    test_configs = [
        (1, 3, 256, 256),   # Your training config
        (1, 3, 384, 384),   # Higher resolution
        (4, 3, 256, 256),   # Batch processing
    ]
    
    for batch_size, channels, height, width in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing configuration: [{batch_size}, {channels}, {height}, {width}]")
        print(f"{'='*60}")
        
        # Create test input
        input_tensor = torch.rand(batch_size, channels, height, width, device=device)
        
        # Benchmark original implementation
        print("Benchmarking original PyTorch implementation...")
        orig_mean, orig_std = benchmark_function(original_gradient_variance, input_tensor)
        print(f"Original: {orig_mean:.3f} ± {orig_std:.3f} ms")
        
        # Benchmark custom implementation (if available)
        if CUSTOM_OP_AVAILABLE:
            print("Benchmarking custom CUDA operator...")
            custom_mean, custom_std = benchmark_function(
                lambda x: custom_gradient_variance(x, 0.6, 0.4), 
                input_tensor
            )
            print(f"Custom:   {custom_mean:.3f} ± {custom_std:.3f} ms")
            
            # Calculate speedup
            speedup = orig_mean / custom_mean
            print(f"Speedup:  {speedup:.2f}x faster")
            
            # Memory usage comparison (approximate)
            # Original creates ~6 intermediate tensors
            # Custom creates 0 intermediate tensors
            orig_memory_mb = 6 * batch_size * channels * height * width * 4 / (1024**2)
            custom_memory_mb = 0
            memory_savings = orig_memory_mb - custom_memory_mb
            print(f"Memory savings: ~{memory_savings:.1f} MB per batch")
        else:
            print("Custom operator not available for benchmarking")

    # End-to-end MobileViT inference benchmark
    print(f"\n{'='*60}")
    print("End-to-end MobileViT inference benchmark")
    print(f"{'='*60}")
    
    from timm import create_model
    model = create_model('mobilevit_s', pretrained=False, num_classes=2).to(device)
    model.eval()
    
    # Test input
    input_tensor = torch.rand(1, 3, 256, 256, device=device)
    
    # Benchmark without custom op
    def mobilevit_original(x):
        return model(x)
    
    orig_infer_mean, _ = benchmark_function(mobilevit_original, input_tensor)
    print(f"MobileViT (original): {orig_infer_mean:.3f} ms")
    
    # Benchmark with custom op
    if CUSTOM_OP_AVAILABLE:
        def mobilevit_custom(x):
            fused = custom_gradient_variance(x, 0.6, 0.4)
            return model(fused)
        
        custom_infer_mean, _ = benchmark_function(mobilevit_custom, input_tensor)
        print(f"MobileViT (custom):   {custom_infer_mean:.3f} ms")
        
        total_speedup = orig_infer_mean / custom_infer_mean
        print(f"Total pipeline speedup: {total_speedup:.2f}x")

if __name__ == "__main__":
    main()
