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
    name="rmsnorm_lib",
    sources=["rmsnorm.cu"],
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
    x: torch.Tensor,
    g: torch.Tensor,
    tag: str,
    y: torch.Tensor,
    N: int = None,
    K: int = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    # RMSNorm benchmark
    y.fill_(0)
    
    # warmup
    if N is not None and K is not None:
        for i in range(warmup):
            perf_func(x, y, g, N, K)
    else:
        for i in range(warmup):
            perf_func(x, g, y)
    
    torch.cuda.synchronize()
    start = time.time()
    
    # iters
    if N is not None and K is not None:
        for i in range(iters):
            perf_func(x, y, g, N, K)
    else:
        for i in range(iters):
            perf_func(x, g, y)
    
    torch.cuda.synchronize()
    end = time.time()
    
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = y.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(y)
    return y, mean_time

def torch_rmsnorm(x, g, eps=1e-5):
    """PyTorch reference implementation of RMSNorm"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    return x_normalized * g

# 修改测试矩阵尺寸配置
sizes = list(range(256, 3073, 256))  # 256到3072，步长256
print(f"Testing matrix sizes: {sizes}")

# 存储性能结果
results = {
    'sizes': [],
    'rmsnorm_f32': [],
    'rmsnorm_f32x4': [],
    'row_rmsnorm_f32': [],
    'torch_rmsnorm': []
}

for size in sizes:
    print("-" * 85)
    print(" " * 40 + f"Matrix Size: {size}x{size}")
    
    N = size
    K = size
    
    # 创建测试数据
    x = torch.randn((N, K)).cuda().float().contiguous()
    g = torch.randn(K).cuda().float().contiguous()  # weight vector
    y = torch.zeros_like(x).cuda().float().contiguous()
    
    # 测试不同的 rmsnorm 核函数
    _, time_f32 = run_benchmark(lib.rmsnorm_f32, x, g, "f32", y, N, K)
    _, time_f32x4 = run_benchmark(lib.rmsnorm_f32x4, x, g, "f32x4", y, N, K)
    _, time_row = run_benchmark(lib.row_rmsnorm_f32_wrapper, x, g, "row_f32", y)
    
    # PyTorch 参考实现
    y_torch = torch.zeros_like(x)
    start = time.time()
    for _ in range(10):  # warmup
        y_torch = torch_rmsnorm(x, g)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(1000):  # benchmark
        y_torch = torch_rmsnorm(x, g)
    torch.cuda.synchronize()
    end = time.time()
    time_torch = (end - start) * 1000 / 1000  # ms per iteration
    
    print(f"{'out_torch':>18}: {y_torch.flatten()[:2].tolist()}, time:{time_torch:.8f}ms")
    
    # 存储结果
    results['sizes'].append(size)
    results['rmsnorm_f32'].append(time_f32)
    results['rmsnorm_f32x4'].append(time_f32x4)
    results['row_rmsnorm_f32'].append(time_row)
    results['torch_rmsnorm'].append(time_torch)
    
    print("-" * 85)

# 绘制性能对比图
plt.figure(figsize=(12, 8))
plt.plot(results['sizes'], results['rmsnorm_f32'], 'o-', label='RMSNorm F32', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['rmsnorm_f32x4'], 's-', label='RMSNorm F32x4', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['row_rmsnorm_f32'], '^-', label='Row RMSNorm F32', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['torch_rmsnorm'], 'd-', label='PyTorch RMSNorm', linewidth=2, markersize=6)

plt.xlabel('Matrix Size (NxN)', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('RMSNorm Performance Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数坐标以便更好地显示差异
plt.tight_layout()

# 保存图表
plt.savefig('/app/LLM-OBE/doc/rmsnorm_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印性能汇总
print("\n" + "="*70)
print("RMSNorm Performance Summary:")
print("="*70)
for i, size in enumerate(results['sizes']):
    print(f"Size {size:4d}: F32={results['rmsnorm_f32'][i]:8.4f}ms, "
          f"F32x4={results['rmsnorm_f32x4'][i]:8.4f}ms, "
          f"Row={results['row_rmsnorm_f32'][i]:8.4f}ms, "
          f"PyTorch={results['torch_rmsnorm'][i]:8.4f}ms")

# 计算加速比
print("\n" + "="*50)
print("Speedup vs PyTorch:")
print("="*50)
for i, size in enumerate(results['sizes']):
    speedup_f32 = results['torch_rmsnorm'][i] / results['rmsnorm_f32'][i]
    speedup_f32x4 = results['torch_rmsnorm'][i] / results['rmsnorm_f32x4'][i]
    speedup_row = results['torch_rmsnorm'][i] / results['row_rmsnorm_f32'][i]
    print(f"Size {size:4d}: F32={speedup_f32:6.2f}x, "
          f"F32x4={speedup_f32x4:6.2f}x, "
          f"Row={speedup_row:6.2f}x")
