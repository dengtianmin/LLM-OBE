# LLM-OBE (Large Language Model - Operator Benchmark & Evaluate)

## 项目概述

LLM-OBE (operation benchmark evaluate)是一个专注于大语言模型核心算子优化的工程项目，通过CUDA并行计算实现高性能算子。这个项目的主要目的是学习gpu模型以及大语言模型相关的知识，记录cuda算子的学习过程，在doc文件夹中给出了算子的性能测试。

本项目参考了以下优秀的开源项目：
- **[LeetCUDA](https://github.com/xlite-dev/LeetCUDA/)**：CUDA算子实现和优化技术
- **[KuiperLLama](https://github.com/zjhellofss/KuiperLLama)**：LLaMA模型实现和架构设计
- **[vLLM](https://github.com/vllm-project/vllm)**：高性能LLM推理引擎
- **[Triton Flash Attention](https://github.com/hkproj/triton-flash-attention)**：注意力机制优化实现

## 🚀 主要特性

- **高性能CUDA算子**：针对GPU架构优化的自定义CUDA内核
- **多精度支持**：支持FP32、FP16等不同精度计算
- **向量化优化**：利用CUDA的向量化指令提升内存带宽利用率
- **内存优化**：共享内存、寄存器优化、双缓冲等高级优化技术
- **性能基准**：完整的性能测试框架和对比分析
- **PyTorch集成**：无缝集成PyTorch生态系统

## 📁 项目结构

```
LLM-OBE/
├── operators/                   # 算子实现
│   ├── cuda/                   # CUDA算子实现
│   │   ├── gemm.py            # GEMM性能测试
│   │   ├── gemm.cu            # GEMM CUDA内核
│   │   ├── gemm_cublas.cu     # cuBLAS接口封装
│   │   ├── rms_norm.py        # RMS归一化测试
│   │   ├── rms_norm.cu        # RMS归一化CUDA内核
│   │   ├── add.py             # 逐元素加法测试
│   │   ├── add.cu             # 逐元素运算CUDA内核
│   │   └── elementwise.cu     # 通用逐元素运算内核
│   └── cpu/                    # CPU参考实现
│       └── rmsnorm.cpp        # RMS归一化CPU实现
└── doc/                        # 文档和性能图表
    ├── *.png                  # 性能对比图表
    └── README.md              # 项目文档
```

## 🔧 支持的算子

### 1. 矩阵乘法 (GEMM)
- **Naive实现**：基础的矩阵乘法实现
- **分块优化**：使用共享内存的分块算法
- **Thread Tile**：增加计算密度的线程分块
- **向量化**：Float4向量化内存访问
- **Bank Conflict优化**：消除共享内存访问冲突
- **双缓冲**：隐藏内存延迟的双缓冲技术
- **cuBLAS集成**：与NVIDIA优化库的性能对比

### 2. RMS归一化 (RMS Normalization)
- **标准实现**：基础的RMS归一化
- **向量化优化**：F32x4向量化处理
- **行级优化**：基于CUB库的行级归约
- **多尺寸支持**：支持256-5120维度的归一化

### 3. 逐元素运算
- **加法运算**：标量和向量化版本
- **多精度支持**：FP32和FP16实现
- **内存对齐优化**：128位内存访问优化



### 我的环境
- CUDA 12.2
- PyTorch 2.7
- Python 3.10
- GCC 11.4

### 快速开始

```python
import torch
from torch.utils.cpp_extension import load

# 加载CUDA算子
lib = load(
    name="sgemm_lib",
    sources=["gemm.cu", "gemm_cublas.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-std=c++17"]
)

# 创建测试矩阵
A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
C = torch.zeros(1024, 1024, device='cuda', dtype=torch.float32)

# 调用自定义GEMM内核
lib.sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf(A, B, C)
```

### 运行性能测试

```bash
# GEMM性能测试
cd operators/cuda
python gemm.py

# RMS归一化性能测试
python rms_norm.py

# 逐元素运算性能测试
python add.py
```


## 📈 性能分析

项目生成的性能图表保存在 `doc/` 目录下：
- `sgemm_performance_comparison.png` - GEMM性能对比
- `rms_norm_performance_comparison.png` - RMS归一化性能对比
- `add_performance_comparison.png` - 加法运算性能对比


