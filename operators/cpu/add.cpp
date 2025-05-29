#include <armadillo>
#include <torch/extension.h>
#include <torch/types.h>

// Element-wise addition using Armadillo
torch::Tensor add_cpu_f32(torch::Tensor a, torch::Tensor b) {
    // Check tensor dimensions
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor dimensions must match");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor b must be float32");
    
    // Get tensor dimensions
    auto shape = a.sizes();
    int rows = shape[0];
    int cols = shape[1];
    
    // Convert PyTorch tensors to Armadillo matrices
    arma::fmat mat_a(static_cast<float*>(a.data_ptr()), rows, cols, false, true);
    arma::fmat mat_b(static_cast<float*>(b.data_ptr()), rows, cols, false, true);
    
    // Perform element-wise addition
    arma::fmat result = mat_a + mat_b;
    
    // Create output tensor
    auto output = torch::zeros_like(a);
    arma::fmat out_mat(static_cast<float*>(output.data_ptr()), rows, cols, false, true);
    out_mat = result;
    
    return output;
}

// Broadcasted addition
torch::Tensor add_broadcast_cpu_f32(torch::Tensor a, torch::Tensor b) {
    // Simple broadcast for vector + scalar case
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");
    
    if (b.numel() == 1) {
        // Scalar broadcast
        auto shape = a.sizes();
        int rows = shape[0];
        int cols = shape[1];
        
        arma::fmat mat_a(static_cast<float*>(a.data_ptr()), rows, cols, false, true);
        float scalar_b = b.item<float>();
        
        arma::fmat result = mat_a + scalar_b;
        
        auto output = torch::zeros_like(a);
        arma::fmat out_mat(static_cast<float*>(output.data_ptr()), rows, cols, false, true);
        out_mat = result;
        
        return output;
    } else {
        // Fallback to element-wise addition
        return add_cpu_f32(a, b);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cpu_f32", &add_cpu_f32, "CPU float32 element-wise addition using Armadillo");
    m.def("add_broadcast_cpu_f32", &add_broadcast_cpu_f32, "CPU float32 broadcasted addition using Armadillo");
}
