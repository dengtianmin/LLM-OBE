import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

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
    g: float,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, out, g)
    else:
        for i in range(warmup):
            _ = perf_func(x, g)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out, g)
    else:
        for i in range(iters):
            out = perf_func(x, g)
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

# PyTorch RMS Norm参考实现
def torch_rms_norm(x, g, eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * g

# 测试支持的K值
Ns = [1, 4, 16]
Ks = [64, 128, 256, 512, 1024]  # 支持的K值
NKs = [(N, K) for N in Ns for K in Ks]

for N, K in NKs:
    print("-" * 85)
    print(" " * 40 + f"N={N}, K={K}")
    
    x = torch.randn((N, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    g = 1.0  # scale factor
    
    try:
        # 测试dispatch版本的kernel
        run_benchmark(lib.rms_norm_f32, x, g, "f32", y)
        
        # 如果K能被4整除，测试f32x4版本
        if K % 4 == 0:
            run_benchmark(lib.rms_norm_f32x4, x, g, "f32x4", y)
        
        # 测试PyTorch参考实现
        run_benchmark(torch_rms_norm, x, g, "torch")
        
    except RuntimeError as e:
        print(f"Error for N={N}, K={K}: {e}")
    
    print("-" * 85)

# 测试row_rmsnorm_f32_wrapper
print("\n" + "="*50)
print("Testing row_rmsnorm_f32_wrapper:")
print("="*50)

for K in [256, 512, 1024, 2048, 4096]:
    print("-" * 85)
    print(" " * 40 + f"Row RMSNorm K={K}")
    
    x = torch.randn((1, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    g = 1.0
    
    try:
        run_benchmark(lib.row_rmsnorm_f32_wrapper, x, g, "row_f32", y)
        run_benchmark(torch_rms_norm, x, g, "torch")
    except RuntimeError as e:
        print(f"Error for K={K}: {e}")
    
    print("-" * 85)

print("\nRMS Norm测试完成!")
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
    y: torch.Tensor,
    g: float,
    tag: str,
    N: int = None,
    K: int = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
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
    x = x * torch.rsqrt(variance + eps)
    return x * g

# 修改测试矩阵尺寸配置
sizes = list(range(256, 2049, 256))  # 256到2048，步长256
print(f"Testing matrix sizes: {sizes}")

# 存储性能结果
results = {
    'sizes': [],
    'rmsnorm_f32': [],
    'rmsnorm_f32x4': [],
    'row_rmsnorm_f32': [],
    'torch_rmsnorm': []
}

g = 1.0  # gamma parameter for RMSNorm

for size in sizes:
    print("-" * 85)
    print(" " * 40 + f"Matrix Size: {size}x{size}")
    
    N, K = size, size
    x = torch.randn((N, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    
    # Test custom RMSNorm kernels
    _, time_f32 = run_benchmark(lib.rmsnorm_f32, x, y, g, "f32", N, K)
    _, time_f32x4 = run_benchmark(lib.rmsnorm_f32x4, x, y, g, "f32x4", N, K)
    _, time_row = run_benchmark(lib.row_rmsnorm_f32_wrapper, x, y, g, "row_f32")
    
    # Test PyTorch reference implementation
    y_torch = torch.zeros_like(x)
    start = time.time()
    for _ in range(10):  # warmup
        y_torch = torch_rmsnorm(x, g)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        y_torch = torch_rmsnorm(x, g)
    torch.cuda.synchronize()
    end = time.time()
    time_torch = (end - start) * 1000 / 1000
    
    out_val = y_torch.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"    out_torch_rmsnorm: {out_val}, time:{time_torch:.8f}ms")
    
    # 存储结果
    results['sizes'].append(size)
    results['rmsnorm_f32'].append(time_f32)
    results['rmsnorm_f32x4'].append(time_f32x4)
    results['row_rmsnorm_f32'].append(time_row)
    results['torch_rmsnorm'].append(time_torch)
    
    print("-" * 85)

# 绘制性能对比图
plt.figure(figsize=(12, 8))
plt.plot(results['sizes'], results['rmsnorm_f32'], 'o-', label='RMSNorm FP32', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['rmsnorm_f32x4'], 's-', label='RMSNorm FP32x4', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['row_rmsnorm_f32'], '^-', label='Row RMSNorm FP32', linewidth=2, markersize=6)
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
print("\n" + "="*60)
print("RMSNorm Performance Summary:")
print("="*60)
for i, size in enumerate(results['sizes']):
    print(f"Size {size:4d}: F32={results['rmsnorm_f32'][i]:8.4f}ms, "
          f"F32x4={results['rmsnorm_f32x4'][i]:8.4f}ms, "
          f"Row={results['row_rmsnorm_f32'][i]:8.4f}ms, "
          f"PyTorch={results['torch_rmsnorm'][i]:8.4f}ms")

# 计算加速比
print("\n" + "="*60)
print("Speedup Analysis (vs PyTorch):")
print("="*60)
for i, size in enumerate(results['sizes']):
    speedup_f32 = results['torch_rmsnorm'][i] / results['rmsnorm_f32'][i]
    speedup_f32x4 = results['torch_rmsnorm'][i] / results['rmsnorm_f32x4'][i]
    speedup_row = results['torch_rmsnorm'][i] / results['row_rmsnorm_f32'][i]
    print(f"Size {size:4d}: F32={speedup_f32:6.2f}x, "
          f"F32x4={speedup_f32x4:6.2f}x, "
          f"Row={speedup_row:6.2f}x")
