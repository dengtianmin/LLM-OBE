#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

// 常量定义
constexpr int WARP_SIZE = 32;
constexpr int kWarpSize = 32;
constexpr int NUM_THREADS = 256;
constexpr int BLOCK_DIM = 256;

// FP32 核函数声明
__global__ void rms_norm_f32_kernel(float *x, float *y, float g, int N, int K);
__global__ void rms_norm_f32x4_kernel(float *x, float *y, float g, int N, int K);
__global__ void row_rmsnorm_f32(float *x, float g, float *y, int K);

// PyTorch绑定函数声明
void rms_norm_f32(torch::Tensor x, torch::Tensor y, float g);
void rms_norm_f32x4(torch::Tensor x, torch::Tensor y, float g);
void row_rms_norm_f32(torch::Tensor x, torch::Tensor y, float g);

