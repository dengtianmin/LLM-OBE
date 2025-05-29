import time
from typing import Optional
import torch
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import numpy as np

torch.set_grad_enabled(False)

# Load the CPU kernels as python modules
try:
    add_lib = load(
        name="add_cpu_lib",
        sources=["add.cpp"],
        extra_cflags=["-std=c++17", "-O3", "-DARMA_DONT_USE_WRAPPER"],
        extra_ldflags=["-larmadillo", "-llapack", "-lblas"],
    )
    print("✓ Add CPU library loaded successfully")
except Exception as e:
    print(f"✗ Failed to load Add CPU library: {e}")
    add_lib = None

try:
    rmsnorm_lib = load(
        name="rmsnorm_cpu_lib", 
        sources=["rmsnorm.cpp"],
        extra_cflags=["-std=c++17", "-O3", "-DARMA_DONT_USE_WRAPPER"],
        extra_ldflags=["-larmadillo", "-llapack", "-lblas"],
    )
    print("✓ RMSNorm CPU library loaded successfully")
except Exception as e:
    print(f"✗ Failed to load RMSNorm CPU library: {e}")
    rmsnorm_lib = None

try:
    gemm_lib = load(
        name="gemm_cpu_lib",
        sources=["gemm.cpp"],
        extra_cflags=["-std=c++17", "-O3", "-DARMA_DONT_USE_WRAPPER"],
        extra_ldflags=["-larmadillo", "-llapack", "-lblas"],
    )
    print("✓ GEMM CPU library loaded successfully")
except Exception as e:
    print(f"✗ Failed to load GEMM CPU library: {e}")
    gemm_lib = None

def run_benchmark(
    perf_func: callable,
    *args,
    tag: str,
    warmup: int = 5,
    iters: int = 100,
    show_all: bool = False,
):
    # Warmup
    for i in range(warmup):
        result = perf_func(*args)
    
    start = time.time()
    for i in range(iters):
        result = perf_func(*args)
    end = time.time()
    
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    
    out_info = f"out_{tag}"
    if hasattr(result, 'flatten'):
        out_val = result.flatten().detach().cpu().numpy().tolist()[:3]
        out_val = [round(v, 8) for v in out_val]
        out_val = [f"{v:<12}" for v in out_val]
        print(f"{out_info:>20}: {out_val}, time:{mean_time:.6f}ms")
    else:
        print(f"{out_info:>20}: time:{mean_time:.6f}ms")
    
    if show_all:
        print(result)
    return result, mean_time

def test_add_operations():
    print("\n" + "="*60)
    print("Testing ADD Operations")
    print("="*60)
    
    sizes = [256, 512, 1024, 2048]
    
    for size in sizes:
        print(f"\nMatrix Size: {size}x{size}")
        print("-" * 40)
        
        a = torch.randn((size, size), dtype=torch.float32)
        b = torch.randn((size, size), dtype=torch.float32)
        
        # PyTorch reference
        _, time_torch = run_benchmark(torch.add, a, b, tag="torch")
        
        if add_lib:
            # Armadillo CPU implementation
            _, time_arma = run_benchmark(add_lib.add_cpu_f32, a, b, tag="arma_cpu")
            
            # Broadcast addition test
            scalar = torch.tensor(2.0, dtype=torch.float32)
            _, time_broadcast = run_benchmark(add_lib.add_broadcast_cpu_f32, a, scalar, tag="arma_broadcast")
            
            print(f"Speedup vs PyTorch: {time_torch/time_arma:.2f}x")

def test_rmsnorm_operations():
    print("\n" + "="*60)
    print("Testing RMSNorm Operations")
    print("="*60)
    
    sizes = [256, 512, 1024, 2048]
    
    for size in sizes:
        print(f"\nMatrix Size: {size}x{size}")
        print("-" * 40)
        
        x = torch.randn((size, size), dtype=torch.float32)
        g = 1.0
        
        # PyTorch reference (using layer norm as approximation)
        def torch_rmsnorm(x, g):
            return torch.nn.functional.layer_norm(x, x.shape[-1:]) * g
        
        _, time_torch = run_benchmark(torch_rmsnorm, x, g, tag="torch_ln")
        
        if rmsnorm_lib:
            # Armadillo CPU implementation
            _, time_arma = run_benchmark(rmsnorm_lib.rmsnorm_cpu_f32, x, g, tag="arma_cpu")
            _, time_naive = run_benchmark(rmsnorm_lib.rmsnorm_naive_cpu_f32, x, g, tag="arma_naive")
            
            print(f"Armadillo vs Naive: {time_naive/time_arma:.2f}x speedup")

def test_gemm_operations():
    print("\n" + "="*60)
    print("Testing GEMM Operations")
    print("="*60)
    
    sizes = [256, 512, 1024]
    
    for size in sizes:
        print(f"\nMatrix Size: {size}x{size}")
        print("-" * 40)
        
        a = torch.randn((size, size), dtype=torch.float32)
        b = torch.randn((size, size), dtype=torch.float32)
        
        # PyTorch reference
        _, time_torch = run_benchmark(torch.mm, a, b, tag="torch")
        
        if gemm_lib:
            # Armadillo CPU implementations
            _, time_arma = run_benchmark(gemm_lib.gemm_cpu_f32, a, b, tag="arma_cpu")
            _, time_blocked = run_benchmark(gemm_lib.gemm_blocked_cpu_f32, a, b, tag="arma_blocked")
            _, time_naive = run_benchmark(gemm_lib.gemm_naive_cpu_f32, a, b, tag="arma_naive")
            
            print(f"Armadillo vs PyTorch: {time_torch/time_arma:.2f}x")
            print(f"Armadillo vs Naive: {time_naive/time_arma:.2f}x speedup")

if __name__ == "__main__":
    print("CPU Operators Test Suite using Armadillo")
    print("=========================================")
    
    # Test individual operations
    if add_lib:
        test_add_operations()
    
    if rmsnorm_lib:
        test_rmsnorm_operations()
    
    if gemm_lib:
        test_gemm_operations()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
