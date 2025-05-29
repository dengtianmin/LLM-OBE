#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define BLOCK_DIM 256
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// -------------------------------------- FP32
// -------------------------------------- Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Block reduce sum device helper for RMS Norm
template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val;
}

// RMS Norm: x: NxK, y': NxK, y'=x/rms(x) each row
template <const int NUM_THREADS = 256>
__global__ void rms_norm_f32_kernel(float *x, float *y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance;                 // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float variance = value * value;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  if (idx < N * K)
    y[idx] = (value * s_variance) * g;
}

// RMS Norm Vec4: x: NxK, y': NxK, y'=x/rms(x) each row
template <const int NUM_THREADS = 256 / 4>
__global__ void rms_norm_f32x4_kernel(float *x, float *y, float g, int N,
                                      int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float4 reg_x = FLOAT4(x[idx]);
  float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y +
                                    reg_x.z * reg_x.z + reg_x.w * reg_x.w)
                                 : 0.0f;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  float4 reg_y;
  reg_y.x = reg_x.x * s_variance * g;
  reg_y.y = reg_x.y * s_variance * g;
  reg_y.z = reg_x.z * s_variance * g;
  reg_y.w = reg_x.w * s_variance * g;
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}

// 新增的row RMS norm核函数
__global__ void row_rmsnorm_f32(float *x, float g, float *y, int K) {
  const float epsilon = 1e-5f;
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = K / pack_size;
  const int pack_off = pack_size * pack_num;

  float variance = 0.0f;
  float4 *x_pack = reinterpret_cast<float4 *>(x);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 x_float4 = *(x_pack + i);
    variance += x_float4.x * x_float4.x;
    variance += x_float4.y * x_float4.y;
    variance += x_float4.z * x_float4.z;
    variance += x_float4.w * x_float4.w;
  }

  for (int i = pack_off + tid; i < K; i += blockDim.x) {
    variance += x[i] * x[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float s_variance;
  variance = BlockReduce(temp).Sum(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / static_cast<float>(K) + epsilon);
  }
  __syncthreads();

  float4 *y_pack = reinterpret_cast<float4 *>(y);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 x_float4 = *(x_pack + i);
    *(y_pack + i) = make_float4(s_variance * x_float4.x * g,
                                s_variance * x_float4.y * g,
                                s_variance * x_float4.z * g,
                                s_variance * x_float4.w * g);
  }

  for (int i = pack_off + tid; i < K; i += blockDim.x) {
    y[i] = g * x[i] * s_variance;
  }
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }

#define LANUCH_RMS_NORM_F32_KERNEL(K)                                          \
  rms_norm_f32_kernel<(K)>                                                     \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F32_KERNEL(N, K)                                     \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
                                                                               \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_RMS_NORM_F32_KERNEL(64)                                             \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_RMS_NORM_F32_KERNEL(128)                                            \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_RMS_NORM_F32_KERNEL(256)                                            \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_RMS_NORM_F32_KERNEL(512)                                            \
    break;                                                                     \
  case 768:                                                                    \
    LANUCH_RMS_NORM_F32_KERNEL(768)                                            \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_RMS_NORM_F32_KERNEL(1024)                                           \
    break;                                                                     \
  default:                                                                     \
    if ((K) <= 1024) {                                                         \
      rms_norm_f32_kernel<1024>                                                \
          <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),           \
                            reinterpret_cast<float *>(y.data_ptr()), g, N, (K)); \
    } else {                                                                    \
      throw std::runtime_error("K too large for F32 kernel, use F32x4 instead"); \
    }                                                                          \
    break;                                                                     \
  }

#define LANUCH_RMS_NORM_F32x4_KERNEL(K)                                        \
  rms_norm_f32x4_kernel<(K) / 4>                                               \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F32x4_KERNEL(N, K)                                   \
  dim3 block((K) / 4);                                                         \
  dim3 grid((N));                                                              \
                                                                               \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_RMS_NORM_F32x4_KERNEL(64) break;                                    \
  case 128:                                                                    \
    LANUCH_RMS_NORM_F32x4_KERNEL(128) break;                                   \
  case 256:                                                                    \
    LANUCH_RMS_NORM_F32x4_KERNEL(256) break;                                   \
  case 512:                                                                    \
    LANUCH_RMS_NORM_F32x4_KERNEL(512) break;                                   \
  case 768:                                                                    \
    LANUCH_RMS_NORM_F32x4_KERNEL(768) break;                                   \
  case 1024:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(1024) break;                                  \
  case 1280:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(1280) break;                                  \
  case 1536:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(1536) break;                                  \
  case 1792:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(1792) break;                                  \
  case 2048:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(2048) break;                                  \
  case 2304:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(2304) break;                                  \
  case 2560:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(2560) break;                                  \
  case 2816:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(2816) break;                                  \
  case 3072:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(3072) break;                                  \
  case 3328:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(3328) break;                                  \
  case 3584:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(3584) break;                                  \
  case 3840:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(3840) break;                                  \
  case 4096:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(4096) break;                                  \
  case 4352:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(4352) break;                                  \
  case 4608:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(4608) break;                                  \
  case 4864:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(4864) break;                                  \
  case 5120:                                                                   \
    LANUCH_RMS_NORM_F32x4_KERNEL(5120) break;                                  \
  default:                                                                     \
    if ((K) % 4 == 0 && (K) / 4 <= 1024) {                                     \
      rms_norm_f32x4_kernel<1024>                                              \
          <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),           \
                            reinterpret_cast<float *>(y.data_ptr()), g, N, (K)); \
    } else {                                                                    \
      throw std::runtime_error("Unsupported K size for F32x4 kernel");         \
    }                                                                          \
    break;                                                                     \
  }

void rms_norm_f32(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F32_KERNEL(N, K)
}

void rms_norm_f32x4(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F32x4_KERNEL(N, K)
}

// 新增PyTorch绑定函数
void row_rms_norm_f32(torch::Tensor x, torch::Tensor y, float g) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  
  dim3 block(BLOCK_DIM);
  dim3 grid(N);
  
  for (int i = 0; i < N; i++) {
    row_rmsnorm_f32<<<1, block>>>(
      reinterpret_cast<float *>(x.data_ptr()) + i * K,
      g,
      reinterpret_cast<float *>(y.data_ptr()) + i * K,
      K
    );
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(rms_norm_f32)
  TORCH_BINDING_COMMON_EXTENSION(rms_norm_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(row_rms_norm_f32)
}