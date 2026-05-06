# test_operator.py
import torch

try:
    # Import your custom operator
    from gradient_variance_cuda import forward
    print("Custom CUDA operator imported successfully!")
    
    # Test on GPU
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.rand(1, 3, 256, 256).cuda()
    print(f"Input shape: {x.shape}")
    
    # Run forward pass
    y = forward(x, 0.6, 0.4)
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        y = forward(x, 0.6, 0.4)
    end.record()
    torch.cuda.synchronize()
    
    avg_time_ms = start.elapsed_time(end) / 100
    print(f"Average latency: {avg_time_ms:.3f} ms")
    print(f"Throughput: {1000/avg_time_ms:.1f} FPS")
    
except ImportError as e:
    print(f"Import error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
    print("Make sure you're running this from the main project directory!")
