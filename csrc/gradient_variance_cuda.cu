// csrc/gradient_variance.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gradient_variance_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    float alpha,
    float beta
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    // Bounds check for total elements
    if (idx >= total_elements) {
        return;
    }
    
    // Calculate indices using integer arithmetic (avoid floating point)
    int temp = idx;
    int w = temp % width;
    temp /= width;
    int h = temp % height;
    temp /= height;
    int c = temp % channels;
    temp /= channels;
    int b = temp;
    
    // Initialize output to zero (handles all cases)
    output[idx] = 0.0f;
    
    // Boundary check: need 2-pixel border for 5x5 window
    // Valid range: h in [2, height-3], w in [2, width-3]
    if (h < 2 || h >= height - 2 || w < 2 || w >= width - 2) {
        return; // Output already set to 0
    }
    
    // Calculate base pointer for current batch and channel
    const long long base_offset = (long long)b * channels * height * width + (long long)c * height * width;
    const float* img_base = input + base_offset;
    
    // === GRADIENT CALCULATION (Sobel 3x3) ===
    // All accesses are within bounds due to boundary check above
    float gx = -img_base[(h-1) * width + (w-1)] 
               + img_base[(h-1) * width + (w+1)]
               - 2.0f * img_base[h * width + (w-1)] 
               + 2.0f * img_base[h * width + (w+1)]
               - img_base[(h+1) * width + (w-1)] 
               + img_base[(h+1) * width + (w+1)];
    
    float gy = -img_base[(h-1) * width + (w-1)] 
               - 2.0f * img_base[(h-1) * width + w] 
               - img_base[(h-1) * width + (w+1)]
               + img_base[(h+1) * width + (w-1)] 
               + 2.0f * img_base[(h+1) * width + w] 
               + img_base[(h+1) * width + (w+1)];
    
    float grad_mag = sqrtf(gx * gx + gy * gy);
    
    // === LOCAL VARIANCE CALCULATION (5x5 window) ===
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const int window_radius = 2;
    
    // Unrolled loop for better performance
    #pragma unroll
    for (int dh = -window_radius; dh <= window_radius; dh++) {
        #pragma unroll
        for (int dw = -window_radius; dw <= window_radius; dw++) {
            int nh = h + dh;
            int nw = w + dw;
            // These are guaranteed to be in bounds due to our boundary check
            float val = img_base[nh * width + nw];
            sum += val;
            sum_sq += val * val;
        }
    }
    
    const float window_area = 25.0f; // 5x5 = 25
    float mean = sum / window_area;
    float variance = (sum_sq / window_area) - (mean * mean);
    
    // Ensure non-negative variance (numerical stability)
    variance = fmaxf(variance, 0.0f);
    
    // === FUSION ===
    output[idx] = alpha * grad_mag + beta * variance;
}

torch::Tensor gradient_variance_cuda_forward(
    torch::Tensor input,
    float alpha,
    float beta
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(input.ndimension() == 4, "Input must be 4D tensor [B, C, H, W]");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    // Create output tensor
    auto output = torch::zeros_like(input);
    
    // Get tensor dimensions
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int total_elements = batch_size * channels * height * width;
    
    // Handle empty tensor case
    if (total_elements == 0) {
        return output;
    }
    
    // Configure kernel launch parameters
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    // Launch kernel
    gradient_variance_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        alpha,
        beta
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    
    // Synchronize to catch runtime errors
    cudaDeviceSynchronize();
    
    return output;
}