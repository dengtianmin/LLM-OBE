#include "add.cuh"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void add_kernel_cu_fp32(int32_t size, const float *in1,
                                   const float *in2, float *out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    float in_val1 = in1[tid];
    float in_val2 = in2[tid];
    out[tid] = in_val1 + in_val2;
  }
}

__global__ void add_kernel_cu_f32x4(int32_t size,  float *in1,
                                     float *in2, float *out) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < size) {
    float4 reg_a = FLOAT4(in1[idx]);
    float4 reg_b = FLOAT4(in2[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(out[idx]) = reg_c;
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

#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_##packed_type(torch::Tensor a, torch::Tensor b,             \
                                 torch::Tensor c) {                            \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    int N = a.numel();                                                          \
    dim3 block(256 / (n_elements));                                            \
    dim3 grid((N / (n_elements) + block.x - 1) / block.x);                     \
    packed_type<<<grid, block>>>(                                              \
        N, reinterpret_cast<element_type *>(a.data_ptr()),                     \
        reinterpret_cast<element_type *>(b.data_ptr()),                        \
        reinterpret_cast<element_type *>(c.data_ptr()));                       \
  }

// CPU版本的add函数
void add_kernel_cpu_wrapper(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    
    float* ptr_a = a.data_ptr<float>();
    float* ptr_b = b.data_ptr<float>();
    float* ptr_c = c.data_ptr<float>();
    int N = a.numel();
    
    for (int i = 0; i < N; i++) {
        ptr_c[i] = ptr_a[i] + ptr_b[i];
    }
}

TORCH_BINDING_ELEM_ADD(add_kernel_cu_fp32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(add_kernel_cu_f32x4, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_kernel_cu_fp32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_kernel_cu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(add_kernel_cpu_wrapper)
}