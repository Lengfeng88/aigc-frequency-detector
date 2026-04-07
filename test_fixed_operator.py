# test_fixed_operator.py
import torch
from gradient_variance_cuda import forward

def test_operator():
    print("Testing fixed custom operator...")
    
    # Test small tensor
    x_small = torch.rand(1, 3, 32, 32).cuda()
    y_small = forward(x_small, 0.6, 0.4)
    print(f"✅ Small tensor [32x32]: {y_small.shape}")
    
    # Test standard size
    x_std = torch.rand(1, 3, 256, 256).cuda()
    y_std = forward(x_std, 0.6, 0.4)
    print(f"✅ Standard tensor [256x256]: {y_std.shape}")
    
    # Test batch size > 1
    x_batch = torch.rand(4, 3, 128, 128).cuda()
    y_batch = forward(x_batch, 0.6, 0.4)
    print(f"✅ Batch tensor [4x128x128]: {y_batch.shape}")
    
    # Test edge case: minimum valid size
    x_min = torch.rand(1, 3, 5, 5).cuda()  # 5x5 is minimum for 5x5 window
    y_min = forward(x_min, 0.6, 0.4)
    print(f"✅ Minimum tensor [5x5]: {y_min.shape}")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    test_operator()