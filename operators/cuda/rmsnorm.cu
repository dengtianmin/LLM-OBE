#include "rmsnorm.cuh"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

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

__global__ void rms_norm_kernel_f32(float *x, float *y, float *g, int N, int K) {
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
  if (idx < N * K) {
    int col_idx = idx % K;  // 计算列索引以获取对应的权重
    y[idx] = (value * s_variance) * g[col_idx];  // 使用权重向量而不是标量
  }
}

__global__ void rms_norm_kernel_f32x4(float *x, float *y, float *g, int N,
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
  if (idx < N * K) {
    int col_idx = idx % K;  // 计算列索引
    float4 reg_g = FLOAT4(g[col_idx]);  // 加载对应的权重向量
    float4 reg_y;
    reg_y.x = reg_x.x * s_variance * reg_g.x;
    reg_y.y = reg_x.y * s_variance * reg_g.y;
    reg_y.z = reg_x.z * s_variance * reg_g.z;
    reg_y.w = reg_x.w * s_variance * reg_g.w;
    FLOAT4(y[idx]) = reg_y;
  }
}

__global__ void row_rmsnorm_f32(float *x, float *g, float *y, int K) {
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

  float4 *g_pack = reinterpret_cast<float4 *>(g);
  float4 *y_pack = reinterpret_cast<float4 *>(y);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 x_float4 = *(x_pack + i);
    float4 g_float4 = *(g_pack + i);
    *(y_pack + i) = make_float4(s_variance * x_float4.x * g_float4.x,
                                s_variance * x_float4.y * g_float4.y,
                                s_variance * x_float4.z * g_float4.z,
                                s_variance * x_float4.w * g_float4.w);
  }

  for (int i = pack_off + tid; i < K; i += blockDim.x) {
    y[i] = g[i] * x[i] * s_variance;
  }
}

// --------------------- PyTorch bindings for custom kernel
// -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// RMSNorm PyTorch wrapper functions
void rmsnorm_f32(torch::Tensor x, torch::Tensor y, torch::Tensor g, int N, int K) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(g, torch::kFloat32)

  dim3 block(NUM_THREADS);
  dim3 grid(N);

  rms_norm_kernel_f32<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                       g.data_ptr<float>(), N, K);
}

void rmsnorm_f32x4(torch::Tensor x, torch::Tensor y, torch::Tensor g, int N, int K) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(g, torch::kFloat32)

  dim3 block(NUM_THREADS);
  dim3 grid(N);

  rms_norm_kernel_f32x4<<<grid, block>>>(x.data_ptr<float>(),
                                         y.data_ptr<float>(), g.data_ptr<float>(), N, K);
}

void row_rmsnorm_f32_wrapper(torch::Tensor x, torch::Tensor g,
                             torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(g, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)

  int K = x.size(-1);
  int N = x.numel() / K;

  dim3 block(BLOCK_DIM);
  dim3 grid(N);

  for (int i = 0; i < N; i++) {
    row_rmsnorm_f32<<<1, block>>>(x.data_ptr<float>() + i * K,
                                  g.data_ptr<float>(),
                                  y.data_ptr<float>() + i * K, K);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(rmsnorm_f32)
  TORCH_BINDING_COMMON_EXTENSION(rmsnorm_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(row_rmsnorm_f32_wrapper)
}
