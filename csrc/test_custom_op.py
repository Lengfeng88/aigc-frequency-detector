# test_custom_op.py
import torch
try:
    from gradient_variance_cuda import forward
    print("✅ Custom operator imported successfully!")
    
    # Test on GPU
    if torch.cuda.is_available():
        x = torch.randn(1, 3, 256, 256).cuda()
        y = forward(x, alpha=0.6, beta=0.4)
        print(f"✅ GPU test passed! Output shape: {y.shape}")
        print(f"✅ Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    else:
        print("⚠️ CUDA not available, skipping GPU test")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()