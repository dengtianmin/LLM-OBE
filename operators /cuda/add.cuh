#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

__global__ void add_kernel_cu_fp32(int32_t size, const float *in1,
    const float *in2, float *out);

__global__ void add_kernel_cu_f32x4(int32_t size, const float *in1,
    const float *in2, float *out);