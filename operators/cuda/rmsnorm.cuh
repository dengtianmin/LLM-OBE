#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cub/cub.cuh>

// 常量定义
constexpr int WARP_SIZE = 32;
constexpr int kWarpSize = 32;
constexpr int NUM_THREADS = 256;
constexpr int BLOCK_DIM = 256;

// 添加核函数声明
__global__ void rms_norm_kernel_f32(float *x, float *y, float *g, int N, int K);
__global__ void rms_norm_kernel_f32x4(float *x, float *y, float *g, int N, int K);
__global__ void row_rmsnorm_f32(float *x, float *g, float *y, int K);

