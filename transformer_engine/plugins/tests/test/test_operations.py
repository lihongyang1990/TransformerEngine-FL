# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Correctness tests for other operations (GEMM, Softmax, Dropout, etc.).
"""

import os
import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/public-nvme/lihongyang/TransformerEngine-FL')

from transformer_engine.plugins.tests.test_utils import (
    get_available_backends,
    get_backend,
    TestCase,
    generate_random_tensor,
)


class OperationsTests(TestCase):
    """Test suite for various operations."""

    def __init__(self, device="cpu"):
        super().__init__(
            "Operations (GEMM, Softmax, Dropout)",
            "Test correctness of GEMM, Softmax, and Dropout operations"
        )
        self.backends = get_available_backends()
        self.device = device

    # =========================================================================
    # GEMM Tests
    # =========================================================================

    def test_gemm_basic(self, M=32, N=64, K=48):
        """Test basic GEMM: C = B @ A (注意顺序！)"""
        print(f"\n  Testing GEMM ({M}x{K}) @ ({K}x{N})")

        # generic_gemm 实际执行 B @ A，所以参数顺序是 (A, transA, B, transB)
        # 但计算的是 output = B @ A
        A = generate_random_tensor((K, N), dtype=torch.float32, device=self.device)  # A 是 K×N
        B = generate_random_tensor((M, K), dtype=torch.float32, device=self.device)  # B 是 M×K
        reference = B @ A  # M×K @ K×N = M×N

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                D = torch.empty((M, N), dtype=torch.float32)
                workspace = torch.empty(1024, dtype=torch.uint8, device=self.device)

                output, _, _, _ = backend.generic_gemm(
                    A, False, B, False, D,
                    None, torch.float32, None, None,
                    False, None, False,
                    workspace, 1024, False, False
                )

                self.assert_close(
                    output, reference, rtol=1e-5, atol=1e-7,
                    msg=f"GEMM output mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_gemm_transpose_a(self, M=32, N=64, K=48):
        """Test GEMM with transposed A: C = B @ A.T"""
        print(f"\n  Testing GEMM transpose A ({N}x{K}).T @ ({M}x{K})")

        # generic_gemm 执行 B @ A.T，参数是 (A, transA=True, B, transB)
        A = generate_random_tensor((N, K), dtype=torch.float32, device=self.device)  # A 是 N×K，转置后是 K×N
        B = generate_random_tensor((M, K), dtype=torch.float32, device=self.device)  # B 是 M×K
        reference = B @ A.T  # M×K @ K×N = M×N

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                D = torch.empty((M, N), dtype=torch.float32)
                workspace = torch.empty(1024, dtype=torch.uint8, device=self.device)

                output, _, _, _ = backend.generic_gemm(
                    A, True, B, False, D,
                    None, torch.float32, None, None,
                    False, None, False,
                    workspace, 1024, False, False
                )

                self.assert_close(
                    output, reference, rtol=1e-5, atol=1e-7,
                    msg=f"GEMM transpose A mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_gemm_3d(self, B=2, M=16, N=32, K=24):
        """Test GEMM with 3D tensors."""
        print(f"\n  Testing 3D GEMM ({B}x{M}x{K}) @ ({K}x{N})")

        A = generate_random_tensor((B, M, K), dtype=torch.float32, device=self.device)
        B_mat = generate_random_tensor((K, N), dtype=torch.float32, device=self.device)
        reference = torch.matmul(A, B_mat)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                D = torch.empty((B, M, N), dtype=torch.float32)
                workspace = torch.empty(1024, dtype=torch.uint8, device=self.device)

                output, _, _, _ = backend.generic_gemm(
                    B_mat, False, A, False, D,
                    None, torch.float32, None, None,
                    False, None, False,
                    workspace, 1024, False, False
                )

                self.assert_close(
                    output, reference, rtol=1e-5, atol=1e-7,
                    msg=f"3D GEMM mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    # =========================================================================
    # Softmax Tests
    # =========================================================================

    def test_scaled_softmax(self, shape=(2, 4, 8)):
        """Test scaled softmax forward."""
        print(f"\n  Testing scaled softmax with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32)
        scale = 0.125
        reference = F.softmax(x * scale, dim=-1)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.scaled_softmax_forward(x, scale)
                self.assert_close(
                    output, reference, rtol=1e-5, atol=1e-7,
                    msg=f"Scaled softmax mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_causal_masked_softmax(self, shape=(2, 4, 8, 8)):
        """Test causal masked softmax."""
        print(f"\n  Testing causal masked softmax with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32)
        scale = 0.125
        seq_len = shape[-1]

        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float32),
            diagonal=1
        )
        reference = F.softmax(x * scale + causal_mask, dim=-1)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.scaled_upper_triang_masked_softmax_forward(x, scale)
                self.assert_close(
                    output, reference, rtol=1e-5, atol=1e-7,
                    msg=f"Causal masked softmax mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    # =========================================================================
    # Dropout Tests
    # =========================================================================

    def test_dropout(self, shape=(4, 8, 16)):
        """Test dropout forward and backward."""
        print(f"\n  Testing dropout with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32)
        dropout_prob = 0.1

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                # Test forward
                output, mask = backend.dropout_fwd(x, dropout_prob)

                # Check that mask is binary
                assert torch.all((mask == 0) | (mask == 1)), \
                    f"Dropout mask should be binary for {backend_name}"

                # Check scaling
                expected_scale = 1.0 / (1.0 - dropout_prob)
                non_zero_vals = output[mask == 1]
                expected_vals = x[mask == 1] * expected_scale
                self.assert_close(
                    non_zero_vals, expected_vals, rtol=1e-6, atol=1e-8,
                    msg=f"Dropout scaling mismatch for {backend_name}"
                )

                # Test backward
                grad_output = generate_random_tensor(shape, dtype=torch.float32)
                grad_input = backend.dropout_bwd(grad_output, mask, dropout_prob)

                expected_grad = grad_output * mask.float() * expected_scale
                self.assert_close(
                    grad_input, expected_grad, rtol=1e-6, atol=1e-8,
                    msg=f"Dropout backward mismatch for {backend_name}"
                )

                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def run_all_tests(self):
        """Run all operation tests."""
        print("\n" + "="*60)
        print("Testing Operations (GEMM, Softmax, Dropout)")
        print("="*60)
        print(f"Available backends: {', '.join(self.backends)}")

        # GEMM tests
        self.test_gemm_basic(M=32, N=64, K=48)
        self.test_gemm_basic(M=64, N=128, K=96)
        self.test_gemm_transpose_a(M=32, N=64, K=48)
        self.test_gemm_3d(B=2, M=16, N=32, K=24)

        # Softmax tests
        self.test_scaled_softmax((2, 4, 8))
        self.test_scaled_softmax((4, 16, 32))
        self.test_causal_masked_softmax((2, 4, 8, 8))
        self.test_causal_masked_softmax((2, 4, 16, 16))

        # Dropout tests
        self.test_dropout((4, 8, 16))
        self.test_dropout((8, 16, 32))

        return self.report()


def main():
    """Run operations tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = OperationsTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
