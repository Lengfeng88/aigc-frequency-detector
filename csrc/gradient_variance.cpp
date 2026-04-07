// csrc/gradient_variance.cpp
#include <torch/extension.h>
#include <vector>

// Forward declarations
torch::Tensor gradient_variance_cuda_forward(
    torch::Tensor input, 
    float alpha, 
    float beta
);

torch::Tensor gradient_variance_cpu_forward(
    torch::Tensor input, 
    float alpha, 
    float beta
);

// Autograd Function
class GradientVarianceFunction : public torch::autograd::Function<GradientVarianceFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        float alpha,
        float beta
    ) {
        ctx->save_for_backward({input});
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["beta"] = beta;
        
        if (input.is_cuda()) {
            return gradient_variance_cuda_forward(input, alpha, beta);
        } else {
            return gradient_variance_cpu_forward(input, alpha, beta);
        }
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto alpha = ctx->saved_data["alpha"].toDouble();
        auto beta = ctx->saved_data["beta"].toDouble();
        auto grad_output = grad_outputs[0];

        // Simple gradient: pass through (you can implement proper backward if needed)
        auto grad_input = torch::zeros_like(input);
        
        return {grad_input, torch::Tensor(), torch::Tensor()};
    }
};

// Public API
torch::Tensor gradient_variance(
    torch::Tensor input,
    float alpha = 0.6f,
    float beta = 0.4f
) {
    TORCH_CHECK(input.ndimension() == 4, "Input must be 4D tensor [B, C, H, W]");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    return GradientVarianceFunction::apply(input, alpha, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gradient_variance, "Gradient-Variance Fusion Feature (with Autograd)");
}