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
    name="sgemm_lib",
    sources=[
        "gemm.cu",
        "gemm_cublas.cu",
    ],
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

MAX_TFLOPS = -1

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    stages: int = -1,
    swizzle: bool = False,
    swizzle_stride: int = 1,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
    collect_stats: bool = False,
):

    global MAX_TFLOPS

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)

    if a.size(0) > 1024 or a.size(1) >= 1024 or b.size(1) > 1024:
        iters = 10

    if swizzle:
        # make swizzle stride as N/4 and multiples of 256
        swizzle_stride = int((int(N / 8) // 256) * 256)
        swizzle_stride = swizzle_stride if swizzle_stride >= 256 else 1
        swizzle = swizzle if swizzle_stride >= 256 else False
    else:
        swizzle_stride = 1  # means no thread block swizzle

    if stages:
        assert swizzle_stride is not None

    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten()[:2].detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}"[:10] for v in out_val]
    TFLOPS = (2 * M * N * K) * 1e-9 / (mean_time)
    mean_time_str = str(f"{mean_time:<12}")[:8]
    swizzle_stride = "NOOP" if swizzle_stride == 1 else swizzle_stride

    if not collect_stats:
        # caculate TFLOPS improved.
        if TFLOPS > MAX_TFLOPS:
            if MAX_TFLOPS > 0:
                improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
                improve = round(improve, 2)
            else:
                improve = 0
            MAX_TFLOPS = TFLOPS
            print(
                f"{out_info:>35}: {out_val}, time:{mean_time_str}ms, "
                f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}(+{improve:.2f}%)"
            )
        else:
            print(
                f"{out_info:>35}: {out_val}, time:{mean_time_str}ms, "
                f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}"
            )
    else:
        print(f"{out_info:>18}: {out_val[:2]}, time:{mean_time:.8f}ms, TFLOPS:{TFLOPS:.4f}")
    
    if show_all:
        print(out)
    return out, mean_time, TFLOPS

# 修改测试矩阵尺寸配置，参考add.py的格式
sizes = list(range(256, 5121, 256))  # 256到5120，步长256
print(f"Testing matrix sizes: {sizes}")

# 存储性能结果
results = {
    'sizes': [],
    'naive': [],
    't8x8bcf': [],
    't8x8dbuf': [],
    'cublas': [],
    'pytorch': []
}

# 预分配内存
MAX_SIZE = max(sizes)
A = torch.randn((MAX_SIZE, MAX_SIZE), dtype=torch.float).cuda()
B = torch.randn((MAX_SIZE, MAX_SIZE), dtype=torch.float).cuda()
C = torch.randn((MAX_SIZE, MAX_SIZE), dtype=torch.float).cuda()
torch.cuda.synchronize()

for size in sizes:
    print("-" * 85)
    print(" " * 40 + f"Matrix Size: {size}x{size}x{size}")
    
    a = A[:size, :size].contiguous()
    b = B[:size, :size].contiguous()
    c = C[:size, :size].contiguous()
    torch.cuda.synchronize()

    # CUDA Cores FP32 tests
    try:
        _, time_naive, tflops_naive = run_benchmark(lib.sgemm_naive_f32, a, b, "naive", c, collect_stats=True)
    except:
        time_naive, tflops_naive = float('inf'), 0
        print(f"{'out_naive':>18}: [SKIP], kernel may not support this size")

    _, time_t8x8bcf, tflops_t8x8bcf = run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4_bcf, a, b, "t8x8bcf", c, collect_stats=True)
    _, time_t8x8dbuf, tflops_t8x8dbuf = run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf, a, b, "t8x8dbuf", c, collect_stats=True)
    _, time_cublas, tflops_cublas = run_benchmark(lib.sgemm_cublas, a, b, "cublas", c, collect_stats=True)
    _, time_pytorch, tflops_pytorch = run_benchmark(partial(torch.matmul, out=c), a, b, "pytorch", collect_stats=True)

    # 存储结果
    results['sizes'].append(size)
    results['naive'].append(time_naive if time_naive != float('inf') else None)
    results['t8x8bcf'].append(time_t8x8bcf)
    results['t8x8dbuf'].append(time_t8x8dbuf)
    results['cublas'].append(time_cublas)
    results['pytorch'].append(time_pytorch)
    
    print("-" * 85)

# 绘制性能对比图
plt.figure(figsize=(12, 8))

# 过滤掉无效数据点
valid_naive_sizes = [s for s, t in zip(results['sizes'], results['naive']) if t is not None]
valid_naive_times = [t for t in results['naive'] if t is not None]

if valid_naive_sizes:
    plt.plot(valid_naive_sizes, valid_naive_times, 'o-', label='SGEMM Naive', linewidth=2, markersize=6)

plt.plot(results['sizes'], results['t8x8bcf'], '^-', label='T8x8 BCF', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['t8x8dbuf'], 'd-', label='T8x8 Double Buffer', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['cublas'], 'v-', label='cuBLAS', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['pytorch'], '<-', label='PyTorch', linewidth=2, markersize=6)

plt.xlabel('Matrix Size (MxNxK)', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('SGEMM Performance Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数坐标以便更好地显示差异
plt.tight_layout()

# 保存图表
plt.savefig('/app/LLM-OBE/doc/sgemm_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印性能汇总
print("\n" + "="*80)
print("Performance Summary:")
print("="*80)
for i, size in enumerate(results['sizes']):
    naive_time = results['naive'][i] if results['naive'][i] is not None else "N/A"
    naive_str = f"{naive_time:8.4f}ms" if isinstance(naive_time, float) else f"{naive_time:>8}"
    
    print(f"Size {size:4d}: Naive={naive_str:>12}, "
          f"T8x8BCF={results['t8x8bcf'][i]:8.4f}ms, "
          f"T8x8DBuf={results['t8x8dbuf'][i]:8.4f}ms, "
          f"cuBLAS={results['cublas'][i]:8.4f}ms, "
          f"PyTorch={results['pytorch'][i]:8.4f}ms")