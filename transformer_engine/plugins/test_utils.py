# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
import numpy as np
from typing import List, Dict, Callable, Any, Optional


def get_available_backends() -> List[str]:
    try:
        from transformer_engine.plugins.transformer_engine_fl.registry import list_backends
        backends = list_backends()
        return [b['name'] for b in backends if b['available']]
    except Exception as e:
        print(f"Warning: Could not load backends: {e}")
        return ['reference']


def get_backend(name: str):
    from transformer_engine.plugins.transformer_engine_fl.registry import get_backend
    return get_backend(name)


def allclose(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def compute_relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    diff = (output - reference).abs()
    relative_error = (diff / (reference.abs() + 1e-10)).mean().item()
    return relative_error


def compute_max_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return (output - reference).abs().max().item()


class TestCase:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: List[str] = []

    def setup(self):
        pass

    def teardown(self):
        pass

    def assert_close(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        msg: str = "",
    ):
        if not allclose(output, reference, rtol, atol):
            max_err = compute_max_error(output, reference)
            rel_err = compute_relative_error(output, reference)
            error_msg = f"{msg}\n  Max error: {max_err:.6e}, Relative error: {rel_err:.6e}"
            self.errors.append(error_msg)
            self.failed += 1
            raise AssertionError(error_msg)
        self.passed += 1

    def report(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"Test: {self.name}")
        if self.description:
            print(f"Description: {self.description}")
        print(f"{'='*60}")
        print(f"Total: {total}, Passed: {self.passed}, Failed: {self.failed}, Skipped: {self.skipped}")
        if self.errors:
            print(f"\nErrors:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        print(f"{'='*60}")
        return self.failed == 0


def generate_random_tensor(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    if dtype in (torch.bfloat16, torch.float16):
        tensor = torch.randn(shape, dtype=torch.float32, device=device)
        tensor = tensor.to(dtype=dtype)
        if requires_grad:
            tensor.requires_grad_(True)
    else:
        tensor = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return tensor


def generate_test_shapes() -> List[tuple]:
    return [
        (2, 4),
        (8, 16),
        (32, 64),
        (2, 4, 8),
        (4, 8, 16),
        (2, 4, 8, 16),
    ]


def run_test_on_backends(
    test_func: Callable,
    backends: Optional[List[str]] = None,
    reference_backend: str = "reference",
) -> Dict[str, bool]:
    if backends is None:
        backends = get_available_backends()

    results = {}
    for backend_name in backends:
        try:
            test_func(backend_name)
            results[backend_name] = True
            print(f"  ✓ {backend_name}")
        except Exception as e:
            results[backend_name] = False
            print(f"  ✗ {backend_name}: {e}")

    return results


def skip_if_backend_unavailable(backend_name: str) -> bool:
    available = get_available_backends()
    return backend_name not in available
