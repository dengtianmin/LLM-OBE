#include <armadillo>
#include <torch/extension.h>
#include <torch/types.h>

// General Matrix Multiplication using Armadillo BLAS backend
torch::Tensor gemm_cpu_f32(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Both tensors must be 2D matrices");
    
    auto a_shape = a.sizes();
    auto b_shape = b.sizes();
    
    int M = a_shape[0];  // rows of A
    int K = a_shape[1];  // cols of A
    int N = b_shape[1];  // cols of B
    
    TORCH_CHECK(K == b_shape[0], "Matrix dimensions don't match for multiplication");
    
    // Convert to Armadillo matrices
    arma::fmat mat_a(static_cast<float*>(a.data_ptr()), M, K, false, true);
    arma::fmat mat_b(static_cast<float*>(b.data_ptr()), K, N, false, true);
    
    // Perform matrix multiplication
    arma::fmat result = mat_a * mat_b;
    
    // Create output tensor
    auto output = torch::zeros({M, N}, torch::kFloat32);
    arma::fmat out_mat(static_cast<float*>(output.data_ptr()), M, N, false, true);
    out_mat = result;
    
    return output;
}

// Block-based matrix multiplication for larger matrices
torch::Tensor gemm_blocked_cpu_f32(torch::Tensor a, torch::Tensor b, int block_size = 64) {
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Both tensors must be 2D matrices");
    
    auto a_shape = a.sizes();
    auto b_shape = b.sizes();
    
    int M = a_shape[0];
    int K = a_shape[1];
    int N = b_shape[1];
    
    TORCH_CHECK(K == b_shape[0], "Matrix dimensions don't match for multiplication");
    
    arma::fmat mat_a(static_cast<float*>(a.data_ptr()), M, K, false, true);
    arma::fmat mat_b(static_cast<float*>(b.data_ptr()), K, N, false, true);
    
    auto output = torch::zeros({M, N}, torch::kFloat32);
    arma::fmat result(static_cast<float*>(output.data_ptr()), M, N, false, true);
    result.zeros();
    
    // Block matrix multiplication
    for (int i = 0; i < M; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < K; k += block_size) {
                int end_i = std::min(i + block_size, M);
                int end_j = std::min(j + block_size, N);
                int end_k = std::min(k + block_size, K);
                
                arma::fmat a_block = mat_a.submat(i, k, end_i - 1, end_k - 1);
                arma::fmat b_block = mat_b.submat(k, j, end_k - 1, end_j - 1);
                
                result.submat(i, j, end_i - 1, end_j - 1) += a_block * b_block;
            }
        }
    }
    
    return output;
}

// Naive matrix multiplication for comparison
torch::Tensor gemm_naive_cpu_f32(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Both tensors must be 2D matrices");
    
    auto a_shape = a.sizes();
    auto b_shape = b.sizes();
    
    int M = a_shape[0];
    int K = a_shape[1];
    int N = b_shape[1];
    
    TORCH_CHECK(K == b_shape[0], "Matrix dimensions don't match for multiplication");
    
    arma::fmat mat_a(static_cast<float*>(a.data_ptr()), M, K, false, true);
    arma::fmat mat_b(static_cast<float*>(b.data_ptr()), K, N, false, true);
    
    auto output = torch::zeros({M, N}, torch::kFloat32);
    arma::fmat result(static_cast<float*>(output.data_ptr()), M, N, false, true);
    
    // Triple nested loop implementation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += mat_a(i, k) * mat_b(k, j);
            }
            result(i, j) = sum;
        }
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_cpu_f32", &gemm_cpu_f32, "CPU float32 matrix multiplication using Armadillo");
    m.def("gemm_blocked_cpu_f32", &gemm_blocked_cpu_f32, "CPU float32 blocked matrix multiplication using Armadillo");
    m.def("gemm_naive_cpu_f32", &gemm_naive_cpu_f32, "CPU float32 naive matrix multiplication using Armadillo");
}
