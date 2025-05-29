#include <armadillo>
#include <torch/extension.h>
#include <torch/types.h>
#include <cmath>

// RMS Norm using Armadillo: y = x / rms(x) * g
torch::Tensor rmsnorm_cpu_f32(torch::Tensor x, float g) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D (N x K)");
    
    auto shape = x.sizes();
    int N = shape[0];  // batch size
    int K = shape[1];  // feature dimension
    
    const float epsilon = 1e-5f;
    
    // Convert to Armadillo matrix
    arma::fmat mat_x(static_cast<float*>(x.data_ptr()), N, K, false, true);
    
    // Create output tensor
    auto output = torch::zeros_like(x);
    arma::fmat mat_y(static_cast<float*>(output.data_ptr()), N, K, false, true);
    
    // Process each row (sample)
    for (int i = 0; i < N; i++) {
        arma::frowvec row = mat_x.row(i);
        
        // Compute RMS: sqrt(mean(x^2))
        float mean_square = arma::mean(arma::square(row));
        float rms = std::sqrt(mean_square + epsilon);
        
        // Normalize and scale
        mat_y.row(i) = (row / rms) * g;
    }
    
    return output;
}

// Naive RMS norm implementation for comparison
torch::Tensor rmsnorm_naive_cpu_f32(torch::Tensor x, float g) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D (N x K)");
    
    auto shape = x.sizes();
    int N = shape[0];
    int K = shape[1];
    
    const float epsilon = 1e-5f;
    
    arma::fmat mat_x(static_cast<float*>(x.data_ptr()), N, K, false, true);
    auto output = torch::zeros_like(x);
    arma::fmat mat_y(static_cast<float*>(output.data_ptr()), N, K, false, true);
    
    for (int i = 0; i < N; i++) {
        arma::frowvec row = mat_x.row(i);
        
        // Calculate variance manually
        float sum_squares = 0.0f;
        for (int j = 0; j < K; j++) {
            sum_squares += row(j) * row(j);
        }
        float mean_square = sum_squares / K;
        float inv_rms = 1.0f / std::sqrt(mean_square + epsilon);
        
        // Apply normalization
        for (int j = 0; j < K; j++) {
            mat_y(i, j) = row(j) * inv_rms * g;
        }
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cpu_f32", &rmsnorm_cpu_f32, "CPU float32 RMS normalization using Armadillo");
    m.def("rmsnorm_naive_cpu_f32", &rmsnorm_naive_cpu_f32, "CPU float32 naive RMS normalization using Armadillo");
}
