#!/usr/bin/env python3
# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Simple standalone benchmark for torch_backend operations.
"""

import torch
import sys
import os
import time
import numpy as np
from datetime import datetime

# Add paths for direct imports
sys.path.insert(0, '/public-nvme/lihongyang/TransformerEngine-FL/transformer_engine/plugins/tex_interface/backends/torch_backend')

# Import cpp_extensions directly
from cpp_extensions import (
    gelu_torch, dgelu_torch,
    relu_torch, drelu_torch,
    silu_torch, dsilu_torch,
    swiglu_torch, dswiglu_torch,
    layernorm_fwd_torch, layernorm_bwd_torch,
    rmsnorm_fwd_torch, rmsnorm_bwd_torch,
    scaled_softmax_forward_torch, scaled_softmax_backward_torch,
    dropout_fwd_torch, dropout_bwd_torch,
    general_gemm_torch,
)


def time_function(func, warmup_iters=10, benchmark_iters=100):
    """Time a function execution."""
    # Warmup
    for _ in range(warmup_iters):
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(benchmark_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        func()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def benchmark_activations(device="cpu"):
    """Benchmark activation functions."""
    print("\n" + "="*70)
    print("Benchmarking Activation Functions")
    print("="*70)

    shapes = [(1024, 1024), (2048, 2048), (4096, 4096)]

    for shape in shapes:
        print(f"\nShape: {shape}")
        x = torch.randn(shape, dtype=torch.float32, device=device)
        grad = torch.randn(shape, dtype=torch.float32, device=device)

        # GELU
        result = time_function(lambda: gelu_torch(x, None))
        print(f"  gelu forward:  {result['mean']:.4f} ± {result['std']:.4f} ms")

        result = time_function(lambda: dgelu_torch(grad, x, None))
        print(f"  gelu backward: {result['mean']:.4f} ± {result['std']:.4f} ms")

        # ReLU
        result = time_function(lambda: relu_torch(x, None))
        print(f"  relu forward:  {result['mean']:.4f} ± {result['std']:.4f} ms")

        result = time_function(lambda: drelu_torch(grad, x, None))
        print(f"  relu backward: {result['mean']:.4f} ± {result['std']:.4f} ms")

        # SiLU
        result = time_function(lambda: silu_torch(x, None))
        print(f"  silu forward:  {result['mean']:.4f} ± {result['std']:.4f} ms")

        result = time_function(lambda: dsilu_torch(grad, x, None))
        print(f"  silu backward: {result['mean']:.4f} ± {result['std']:.4f} ms")


def benchmark_normalization(device="cpu"):
    """Benchmark normalization functions."""
    print("\n" + "="*70)
    print("Benchmarking Normalization Functions")
    print("="*70)

    shapes = [(8, 512, 768), (16, 512, 1024), (32, 512, 2048)]
    eps = 1e-5

    for shape in shapes:
        print(f"\nShape: {shape}")
        hidden_size = shape[-1]
        x = torch.randn(shape, dtype=torch.float32, device=device)
        weight = torch.ones(hidden_size, dtype=torch.float32, device=device)
        bias = torch.zeros(hidden_size, dtype=torch.float32, device=device)
        grad_output = torch.randn(shape, dtype=torch.float32, device=device)

        # LayerNorm
        result = time_function(lambda: layernorm_fwd_torch(x, weight, bias, eps, None, None, torch.float32, 0, False))
        print(f"  layernorm forward:  {result['mean']:.4f} ± {result['std']:.4f} ms")

        output, mean, rsigma = layernorm_fwd_torch(x, weight, bias, eps, None, None, torch.float32, 0, False)
        result = time_function(lambda: layernorm_bwd_torch(grad_output, x, mean, rsigma, weight, 0, False))
        print(f"  layernorm backward: {result['mean']:.4f} ± {result['std']:.4f} ms")

        # RMSNorm
        result = time_function(lambda: rmsnorm_fwd_torch(x, weight, eps, None, None, torch.float32, 0, False))
        print(f"  rmsnorm forward:    {result['mean']:.4f} ± {result['std']:.4f} ms")

        output, _, rsigma = rmsnorm_fwd_torch(x, weight, eps, None, None, torch.float32, 0, False)
        result = time_function(lambda: rmsnorm_bwd_torch(grad_output, x, rsigma, weight, 0, False, eps))
        print(f"  rmsnorm backward:   {result['mean']:.4f} ± {result['std']:.4f} ms")


def benchmark_gemm(device="cpu"):
    """Benchmark GEMM operations."""
    print("\n" + "="*70)
    print("Benchmarking GEMM Operations")
    print("="*70)

    configs = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

    for M, N, K in configs:
        print(f"\nConfig: M={M}, N={N}, K={K}")
        A = torch.randn(M, K, dtype=torch.float32, device=device)
        B = torch.randn(K, N, dtype=torch.float32, device=device)
        D = torch.empty(M, N, dtype=torch.float32, device=device)
        workspace = torch.empty(1024, dtype=torch.uint8, device=device)

        result = time_function(lambda: general_gemm_torch(
            A, False, B, False, D,
            None, torch.float32, None, None,
            False, None, False,
            workspace, 1024, False, False
        ))

        # Compute GFLOPS
        flops = 2 * M * N * K
        gflops = (flops / 1e9) / (result['mean'] / 1000)

        print(f"  gemm: {result['mean']:.4f} ± {result['std']:.4f} ms ({gflops:.2f} GFLOPS)")


def benchmark_softmax(device="cpu"):
    """Benchmark softmax operations."""
    print("\n" + "="*70)
    print("Benchmarking Softmax Operations")
    print("="*70)

    shapes = [(8, 512, 512), (16, 1024, 1024), (32, 2048, 2048)]
    scale = 1.0

    for shape in shapes:
        print(f"\nShape: {shape}")
        x = torch.randn(shape, dtype=torch.float32, device=device)
        grad = torch.randn(shape, dtype=torch.float32, device=device)

        # Forward
        result = time_function(lambda: scaled_softmax_forward_torch(x, scale))
        print(f"  softmax forward:  {result['mean']:.4f} ± {result['std']:.4f} ms")

        # Backward
        output = scaled_softmax_forward_torch(x, scale)
        result = time_function(lambda: scaled_softmax_backward_torch(grad, output, scale))
        print(f"  softmax backward: {result['mean']:.4f} ± {result['std']:.4f} ms")


def benchmark_dropout(device="cpu"):
    """Benchmark dropout operations."""
    print("\n" + "="*70)
    print("Benchmarking Dropout Operations")
    print("="*70)

    shapes = [(1024, 1024), (2048, 2048), (4096, 4096)]
    dropout_prob = 0.1

    for shape in shapes:
        print(f"\nShape: {shape}")
        x = torch.randn(shape, dtype=torch.float32, device=device)
        grad = torch.randn(shape, dtype=torch.float32, device=device)

        # Forward
        result = time_function(lambda: dropout_fwd_torch(x, dropout_prob))
        print(f"  dropout forward:  {result['mean']:.4f} ± {result['std']:.4f} ms")

        # Backward
        output, mask = dropout_fwd_torch(x, dropout_prob)
        result = time_function(lambda: dropout_bwd_torch(grad, mask, dropout_prob))
        print(f"  dropout backward: {result['mean']:.4f} ± {result['std']:.4f} ms")


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print(" "*20 + "TEX Interface Benchmarks")
    print("="*70)

    # Detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nDevice: CUDA - {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print(f"\nDevice: CPU")
    print(f"PyTorch Version: {torch.__version__}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Timestamp: {timestamp}")

    # Run benchmarks
    benchmark_activations(device)
    benchmark_normalization(device)
    benchmark_gemm(device)
    benchmark_softmax(device)
    benchmark_dropout(device)

    print("\n" + "="*70)
    print("Benchmark Complete")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
