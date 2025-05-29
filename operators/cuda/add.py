import time
from functools import partial
from typing import Optional
import torch
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import numpy as np

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="add_lib",
    sources=["add.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    # torch.dot vs custom dot_prod kernel
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

# 修改测试矩阵尺寸配置
sizes = list(range(256, 5121, 256))  # 256到5120，步长256
print(f"Testing matrix sizes: {sizes}")

# 存储性能结果
results = {
    'sizes': [],
    'f32': [],
    'f32x4': [],
    'f32_th': [],
    'f32_cpu': []
}

for size in sizes:
    print("-" * 85)
    print(" " * 40 + f"Matrix Size: {size}x{size}")
    
    a = torch.randn((size, size)).cuda().float().contiguous()
    b = torch.randn((size, size)).cuda().float().contiguous()
    c = torch.zeros_like(a).cuda().float().contiguous()
    
    # GPU版本测试
    _, time_f32 = run_benchmark(lib.elementwise_add_kernel_cu_fp32, a, b, "f32", c)
    _, time_f32x4 = run_benchmark(lib.elementwise_add_kernel_cu_f32x4, a, b, "f32x4", c)
    _, time_f32_th = run_benchmark(partial(torch.add, out=c), a, b, "f32_th")
    
    # CPU版本测试
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_cpu = torch.zeros_like(a_cpu)
    _, time_f32_cpu = run_benchmark(lib.add_kernel_cpu_wrapper, a_cpu, b_cpu, "f32_cpu", c_cpu)
    
    # 存储结果
    results['sizes'].append(size)
    results['f32'].append(time_f32)
    results['f32x4'].append(time_f32x4)
    results['f32_th'].append(time_f32_th)
    results['f32_cpu'].append(time_f32_cpu)
    
    print("-" * 85)

# 绘制性能对比图
plt.figure(figsize=(12, 8))
plt.plot(results['sizes'], results['f32'], 'o-', label='FP32', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['f32x4'], 's-', label='FP32x4', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['f32_th'], '^-', label='PyTorch FP32', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['f32_cpu'], 'd-', label='CPU FP32', linewidth=2, markersize=6)

plt.xlabel('Matrix Size (NxN)', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('Matrix Addition Performance Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数坐标以便更好地显示差异
plt.tight_layout()

# 保存图表
plt.savefig('/app/LLM-OBE/doc/add_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印性能汇总
print("\n" + "="*50)
print("Performance Summary:")
print("="*50)
for i, size in enumerate(results['sizes']):
    print(f"Size {size:4d}: F32={results['f32'][i]:8.4f}ms, "
          f"F32x4={results['f32x4'][i]:8.4f}ms, "
          f"PyTorch={results['f32_th'][i]:8.4f}ms, "
          f"CPU={results['f32_cpu'][i]:8.4f}ms")