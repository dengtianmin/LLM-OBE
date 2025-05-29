import time
from typing import Optional
import torch
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import numpy as np

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="rms_norm_lib",
    sources=["rms_norm.cu"],
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


# un-fused naive rms norm
def naive_rms_norm(x: torch.Tensor, g: float):
    # y'=x/rms(x) 1/rms(x) = rsqrtf(sum(x^2)/K)
    s_rms = torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True))
    y = (x * s_rms) * g
    return y


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    g = 1.0
    if out is not None:
        out.fill_(0)
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
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>17}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


# 修改测试矩阵尺寸配置，使用指定的K值列表
sizes = [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120]
print(f"Testing matrix sizes: {sizes}")

# 存储性能结果
results = {
    'sizes': [],
    'f32': [],
    'f32x4': [],
    'row_f32': [],
    'f32_th': []
}

for size in sizes:
    print("-" * 85)
    print(" " * 40 + f"Matrix Size: {size}x{size}")
    
    x = torch.randn((size, size)).cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    
    # 测试不同的RMS norm实现
    try:
        if size <= 1024:
            _, time_f32 = run_benchmark(lib.rms_norm_f32, x, "f32", out)
        else:
            time_f32 = float('inf')  # 对于大尺寸，F32版本不支持
            print(f"{'out_f32':>17}: [N/A], F32 kernel not supported for K>{1024}")
    except Exception as e:
        time_f32 = float('inf')
        print(f"{'out_f32':>17}: [ERROR], {str(e)}")
    
    _, time_f32x4 = run_benchmark(lib.rms_norm_f32x4, x, "f32x4", out)
    _, time_row_f32 = run_benchmark(lib.row_rms_norm_f32, x, "row_f32", out)
    _, time_f32_th = run_benchmark(naive_rms_norm, x, "f32_th")
    
    # 存储结果
    results['sizes'].append(size)
    results['f32'].append(time_f32 if time_f32 != float('inf') else None)
    results['f32x4'].append(time_f32x4)
    results['row_f32'].append(time_row_f32)
    results['f32_th'].append(time_f32_th)
    
    print("-" * 85)

# 绘制性能对比图
plt.figure(figsize=(12, 8))

# 过滤掉无效的F32数据点
valid_f32_sizes = [s for s, t in zip(results['sizes'], results['f32']) if t is not None]
valid_f32_times = [t for t in results['f32'] if t is not None]

if valid_f32_sizes:
    plt.plot(valid_f32_sizes, valid_f32_times, 'o-', label='RMS Norm F32', linewidth=2, markersize=6)

plt.plot(results['sizes'], results['f32x4'], 's-', label='RMS Norm F32x4', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['row_f32'], '^-', label='Row RMS Norm F32', linewidth=2, markersize=6)
plt.plot(results['sizes'], results['f32_th'], 'd-', label='PyTorch RMS Norm', linewidth=2, markersize=6)

plt.xlabel('Matrix Size (NxN)', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('RMS Norm Performance Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数坐标以便更好地显示差异
plt.tight_layout()

# 保存图表
plt.savefig('/app/LLM-OBE/doc/rms_norm_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印性能汇总
print("\n" + "="*50)
print("Performance Summary:")
print("="*50)
for i, size in enumerate(results['sizes']):
    f32_time = results['f32'][i] if results['f32'][i] is not None else "N/A"
    f32_str = f"{f32_time:8.4f}ms" if isinstance(f32_time, float) else f32_time
    print(f"Size {size:4d}: F32={f32_str:>12}, "
          f"F32x4={results['f32x4'][i]:8.4f}ms, "
          f"Row F32={results['row_f32'][i]:8.4f}ms, "
          f"PyTorch={results['f32_th'][i]:8.4f}ms")