// csrc/gradient_variance_cpu.cpp
#include <torch/extension.h>
#include <cmath>

torch::Tensor gradient_variance_cpu_forward(
    torch::Tensor input,
    float alpha,
    float beta
) {
    auto output = torch::zeros_like(input);
    auto input_acc = input.accessor<float, 4>();
    auto output_acc = output.accessor<float, 4>();
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 2; h < height - 2; h++) {
                for (int w = 2; w < width - 2; w++) {
                    // Sobel gradient
                    float gx = -input_acc[b][c][h-1][w-1] + input_acc[b][c][h-1][w+1]
                             - 2*input_acc[b][c][h][w-1] + 2*input_acc[b][c][h][w+1]
                             - input_acc[b][c][h+1][w-1] + input_acc[b][c][h+1][w+1];
                    
                    float gy = -input_acc[b][c][h-1][w-1] - 2*input_acc[b][c][h-1][w] - input_acc[b][c][h-1][w+1]
                             + input_acc[b][c][h+1][w-1] + 2*input_acc[b][c][h+1][w] + input_acc[b][c][h+1][w+1];
                    
                    float grad_mag = std::sqrt(gx*gx + gy*gy);
                    
                    // Local variance (5x5)
                    float sum = 0.0f, sum_sq = 0.0f;
                    int count = 0;
                    for (int dh = -2; dh <= 2; dh++) {
                        for (int dw = -2; dw <= 2; dw++) {
                            float val = input_acc[b][c][h+dh][w+dw];
                            sum += val;
                            sum_sq += val * val;
                            count++;
                        }
                    }
                    float mean = sum / count;
                    float variance = (sum_sq / count) - (mean * mean);
                    variance = std::max(variance, 0.0f);
                    
                    output_acc[b][c][h][w] = alpha * grad_mag + beta * variance;
                }
            }
        }
    }
    
    return output;
}