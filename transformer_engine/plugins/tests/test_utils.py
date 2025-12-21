# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Utilities for testing TEX Interface backends.
"""

import torch
import numpy as np
from typing import List, Dict, Callable, Any, Optional


def get_available_backends() -> List[str]:
    """Get list of available backend names."""
    try:
        import sys
        sys.path.insert(0, '/public-nvme/lihongyang/TransformerEngine-FL')
        from transformer_engine.plugins.tex_interface.registry import list_backends
        backends = list_backends()
        return [b['name'] for b in backends if b['available']]
    except Exception as e:
        print(f"Warning: Could not load backends: {e}")
        return ['torch']  # Fallback to torch backend


def get_backend(name: str):
    """Get backend instance by name."""
    import sys
    sys.path.insert(0, '/public-nvme/lihongyang/TransformerEngine-FL')
    from transformer_engine.plugins.tex_interface.registry import get_backend
    return get_backend(name)


def allclose(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if two tensors are close within tolerance.

    Args:
        a: First tensor
        b: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if tensors are close
    """
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def compute_relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Compute relative error between output and reference.

    Args:
        output: Output tensor
        reference: Reference tensor

    Returns:
        Relative error as float
    """
    diff = (output - reference).abs()
    relative_error = (diff / (reference.abs() + 1e-10)).mean().item()
    return relative_error


def compute_max_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Compute maximum absolute error.

    Args:
        output: Output tensor
        reference: Reference tensor

    Returns:
        Maximum error as float
    """
    return (output - reference).abs().max().item()


class TestCase:
    """Base class for test cases."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: List[str] = []

    def setup(self):
        """Setup before each test."""
        pass

    def teardown(self):
        """Cleanup after each test."""
        pass

    def assert_close(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        msg: str = "",
    ):
        """Assert that two tensors are close."""
        if not allclose(output, reference, rtol, atol):
            max_err = compute_max_error(output, reference)
            rel_err = compute_relative_error(output, reference)
            error_msg = f"{msg}\n  Max error: {max_err:.6e}, Relative error: {rel_err:.6e}"
            self.errors.append(error_msg)
            self.failed += 1
            raise AssertionError(error_msg)
        self.passed += 1

    def report(self):
        """Print test report."""
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
    """Generate random tensor for testing."""
    tensor = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return tensor


def generate_test_shapes() -> List[tuple]:
    """Generate common test shapes."""
    return [
        (2, 4),          # Small 2D
        (8, 16),         # Medium 2D
        (32, 64),        # Large 2D
        (2, 4, 8),       # Small 3D
        (4, 8, 16),      # Medium 3D
        (2, 4, 8, 16),   # 4D
    ]


def run_test_on_backends(
    test_func: Callable,
    backends: Optional[List[str]] = None,
    reference_backend: str = "torch",
) -> Dict[str, bool]:
    """
    Run a test function on multiple backends.

    Args:
        test_func: Test function that takes backend name as argument
        backends: List of backend names to test (None = all available)
        reference_backend: Backend to use as reference

    Returns:
        Dictionary mapping backend name to test result (True=pass, False=fail)
    """
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
    """Check if backend should be skipped."""
    available = get_available_backends()
    return backend_name not in available
