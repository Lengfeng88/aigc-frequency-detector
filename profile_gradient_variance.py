# profile_gradient_variance.py
import torch
import torch.nn.functional as F
import time
from torch.profiler import profile, record_function, ProfilerActivity

def original_gradient_variance(img, alpha=0.6, beta=0.4):
    """
    原生 PyTorch 实现：局部梯度-方差融合特征
    输入: [B, C, H, W] RGB 图像 (0-1 归一化)
    输出: [B, C, H, W] 融合特征
    """
    B, C, H, W = img.shape
    
    # Sobel 算子 (3x3)
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
    
    # 扩展到多通道 (groups=C)
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)
    
    # 计算梯度
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # 避免除零
    
    # 局部方差 (5x5 window)
    # 先计算局部均值
    local_mean = F.avg_pool2d(img, kernel_size=5, stride=1, padding=2)
    # 再计算局部平方均值
    local_mean_sq = F.avg_pool2d(img**2, kernel_size=5, stride=1, padding=2)
    # 方差 = E[X^2] - (E[X])^2
    local_var = torch.clamp(local_mean_sq - local_mean**2, min=0)
    
    # 融合
    fused = alpha * grad_mag + beta * local_var
    return fused

def benchmark_latency():
    """测量平均延迟"""
    print("=== 延迟基准测试 ===")
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
    
    print(f"输入尺寸: {img.shape}")
    print(f"平均延迟: {avg_time:.2f} ms")
    print(f"吞吐量: {1000/avg_time:.2f} FPS\n")

def profile_operations():
    """详细性能分析"""
    print("=== 详细性能分析 (PyTorch Profiler) ===")
    img = torch.rand(1, 3, 256, 256).cuda()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with record_function("gradient_variance"):
            out = original_gradient_variance(img)
    
    # 打印关键指标
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        max_name_column_width=50,
        row_limit=10
    ))
    
    # 保存 trace 文件（用于 Chrome://tracing）
    prof.export_chrome_trace("gradient_variance_trace.json")
    print("\n已保存 trace 文件: gradient_variance_trace.json")
    print("用 Chrome 打开 chrome://tracing 加载查看详细 timeline")

if __name__ == "__main__":
    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备运行此脚本")
    
    print("🔍 空间域局部不一致性检测 - 性能分析")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}\n")
    
    benchmark_latency()
    profile_operations()