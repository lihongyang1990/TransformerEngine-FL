# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Utilities for benchmarking TEX Interface backends.
"""

import torch
import time
import numpy as np
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    backend_name: str
    operation_name: str
    shape: tuple
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_gflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None


class Benchmark:
    """Benchmark runner for operations."""

    def __init__(
        self,
        name: str,
        warmup_iters: int = 10,
        benchmark_iters: int = 100,
    ):
        self.name = name
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.results: List[BenchmarkResult] = []

    def time_function(
        self,
        func: Callable,
        *args,
        sync: bool = True,
        **kwargs
    ) -> List[float]:
        """
        Time a function execution.

        Args:
            func: Function to time
            *args: Positional arguments to func
            sync: Whether to synchronize CUDA before timing
            **kwargs: Keyword arguments to func

        Returns:
            List of execution times in milliseconds
        """
        times = []

        # Warmup
        for _ in range(self.warmup_iters):
            func(*args, **kwargs)
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()

        # Benchmark
        for _ in range(self.benchmark_iters):
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            func(*args, **kwargs)

            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return times

    def compute_flops(self, operation: str, *args) -> Optional[float]:
        """
        Compute FLOPs for an operation.

        Args:
            operation: Operation name
            *args: Operation-specific arguments

        Returns:
            Number of FLOPs, or None if not computable
        """
        if operation == "gemm":
            M, N, K = args
            return 2 * M * N * K  # Multiply-add

        elif operation == "layernorm":
            shape = args[0]
            total_elements = np.prod(shape)
            hidden_size = shape[-1]
            # Mean, variance, normalize, scale, shift
            return total_elements * (3 + 2 * hidden_size)

        elif operation == "rmsnorm":
            shape = args[0]
            total_elements = np.prod(shape)
            hidden_size = shape[-1]
            # RMS, normalize, scale
            return total_elements * (2 + hidden_size)

        elif operation == "softmax":
            shape = args[0]
            total_elements = np.prod(shape)
            # Exp, sum, divide
            return total_elements * 3

        elif operation in ["gelu", "relu", "silu"]:
            shape = args[0]
            total_elements = np.prod(shape)
            # Assume 5 ops per element for GELU
            ops_per_element = 5 if operation == "gelu" else 1
            return total_elements * ops_per_element

        return None

    def compute_bandwidth(self, operation: str, *args) -> Optional[float]:
        """
        Compute memory bandwidth requirement in GB.

        Args:
            operation: Operation name
            *args: Operation-specific arguments

        Returns:
            Memory bandwidth in GB, or None if not computable
        """
        bytes_per_element = 4  # Assuming float32

        if operation == "gemm":
            M, N, K = args
            # Read A (MxK), B (KxN), Write C (MxN)
            return (M*K + K*N + M*N) * bytes_per_element / 1e9

        elif operation in ["layernorm", "rmsnorm"]:
            shape = args[0]
            total_elements = np.prod(shape)
            # Read input, weight, write output, mean, variance
            return total_elements * 5 * bytes_per_element / 1e9

        elif operation == "softmax":
            shape = args[0]
            total_elements = np.prod(shape)
            # Read input, write output
            return total_elements * 2 * bytes_per_element / 1e9

        elif operation in ["gelu", "relu", "silu"]:
            shape = args[0]
            total_elements = np.prod(shape)
            # Read input, write output
            return total_elements * 2 * bytes_per_element / 1e9

        return None

    def benchmark_operation(
        self,
        backend_name: str,
        operation_name: str,
        func: Callable,
        shape: tuple,
        flops_args: Optional[tuple] = None,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a single operation.

        Args:
            backend_name: Name of the backend
            operation_name: Name of the operation
            func: Function to benchmark
            shape: Shape of input tensors
            flops_args: Arguments for FLOPS computation
            *args: Arguments to func
            **kwargs: Keyword arguments to func

        Returns:
            BenchmarkResult object
        """
        times = self.time_function(func, *args, **kwargs)

        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        # Compute throughput
        throughput_gflops = None
        if flops_args:
            flops = self.compute_flops(operation_name, *flops_args)
            if flops:
                throughput_gflops = (flops / 1e9) / (mean_time / 1000)

        # Compute bandwidth
        bandwidth_gbps = None
        if flops_args:
            bandwidth_gb = self.compute_bandwidth(operation_name, *flops_args)
            if bandwidth_gb:
                bandwidth_gbps = bandwidth_gb / (mean_time / 1000)

        result = BenchmarkResult(
            backend_name=backend_name,
            operation_name=operation_name,
            shape=shape,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            throughput_gflops=throughput_gflops,
            bandwidth_gbps=bandwidth_gbps,
        )

        self.results.append(result)
        return result

    def print_results(self):
        """Print benchmark results in a formatted table."""
        if not self.results:
            print("No benchmark results to display")
            return

        print(f"\n{'='*100}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*100}")

        # Group results by operation and shape
        from collections import defaultdict
        grouped = defaultdict(list)
        for result in self.results:
            key = (result.operation_name, result.shape)
            grouped[key].append(result)

        # Print each group
        for (operation, shape), group in grouped.items():
            print(f"\nOperation: {operation}, Shape: {shape}")
            print(f"{'-'*100}")
            print(f"{'Backend':<15} {'Mean(ms)':<12} {'Std(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'GFLOPS':<12} {'GB/s':<12}")
            print(f"{'-'*100}")

            for result in sorted(group, key=lambda x: x.mean_time_ms):
                gflops_str = f"{result.throughput_gflops:.2f}" if result.throughput_gflops else "N/A"
                bandwidth_str = f"{result.bandwidth_gbps:.2f}" if result.bandwidth_gbps else "N/A"

                print(
                    f"{result.backend_name:<15} "
                    f"{result.mean_time_ms:<12.4f} "
                    f"{result.std_time_ms:<12.4f} "
                    f"{result.min_time_ms:<12.4f} "
                    f"{result.max_time_ms:<12.4f} "
                    f"{gflops_str:<12} "
                    f"{bandwidth_str:<12}"
                )

        print(f"{'='*100}\n")

    def save_results_csv(self, filename: str):
        """Save benchmark results to CSV file."""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Backend', 'Operation', 'Shape', 'Mean(ms)', 'Std(ms)',
                'Min(ms)', 'Max(ms)', 'GFLOPS', 'GB/s'
            ])

            for result in self.results:
                writer.writerow([
                    result.backend_name,
                    result.operation_name,
                    str(result.shape),
                    f"{result.mean_time_ms:.4f}",
                    f"{result.std_time_ms:.4f}",
                    f"{result.min_time_ms:.4f}",
                    f"{result.max_time_ms:.4f}",
                    f"{result.throughput_gflops:.2f}" if result.throughput_gflops else "N/A",
                    f"{result.bandwidth_gbps:.2f}" if result.bandwidth_gbps else "N/A",
                ])

        print(f"Results saved to {filename}")
