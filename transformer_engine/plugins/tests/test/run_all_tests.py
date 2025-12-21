#!/usr/bin/env python3
# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Run all correctness tests for TEX Interface backends.
"""

import sys
import torch

sys.path.insert(0, '/public-nvme/lihongyang/TransformerEngine-FL')

from test_activations import ActivationTests
from test_normalization import NormalizationTests
from test_operations import OperationsTests


def main():
    """Run all test suites."""
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*70)
    print(" "*15 + "TEX Interface Backend Tests")
    print("="*70)
    print(f"Using device: {device}\n")

    test_suites = [
        ActivationTests(device=device),
        NormalizationTests(device=device),
        OperationsTests(device=device),
    ]

    results = []
    for suite in test_suites:
        success = suite.run_all_tests()
        results.append((suite.name, success))

    # Print summary
    print("\n" + "="*70)
    print(" "*25 + "Test Summary")
    print("="*70)

    total_passed = sum(1 for _, success in results if success)
    total_tests = len(results)

    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name:40s} {status}")

    print("="*70)
    print(f"Total: {total_passed}/{total_tests} test suites passed")
    print("="*70)

    return 0 if all(success for _, success in results) else 1


if __name__ == "__main__":
    exit(main())
