# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import sys

import torch


def test_fallback_mechanism():
    print("\n" + "=" * 60)
    print("Testing Fallback Mechanism")
    print("=" * 60 + "\n")

    from transformer_engine.plugins.transformer_engine_fl import get_backend, list_backends

    print("Available backends:")
    for b in list_backends():
        print(f"  - {b['name']}: priority={b['priority']}, available={b.get('available', 'unknown')}")

    print("\n--- Testing FlagOS Backend ---")
    flagos = get_backend("flagos")
    print(f"Got backend: {flagos.name}")

    print("\n[Test 1] Testing implemented method: get_cublasLt_version()")
    try:
        version = flagos.get_cublasLt_version()
        print(f"  [OK] get_cublasLt_version() = {version}")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")

    print("\n[Test 2] Testing unimplemented method (inherited from base): quantize()")
    try:
        result = flagos.quantize(None, None)
        print(f"  [UNEXPECTED] Got result: {result}")
    except NotImplementedError as e:
        print(f"  [OK] Got expected NotImplementedError")
    except Exception as e:
        print(f"  [INFO] Got exception: {type(e).__name__}: {e}")

    print("\n[Test 3] Testing reference backend for fallback target")
    try:
        reference = get_backend("reference")
        print(f"  [OK] Got reference backend: {reference.name}")

        import inspect
        gelu_method = getattr(reference, 'gelu', None)
        if gelu_method:
            source = inspect.getsourcefile(type(reference).gelu)
            print(f"  [INFO] gelu method source: {source}")
    except Exception as e:
        print(f"  [INFO] {type(e).__name__}: {e}")

    print("\n[Test 4] Checking method inheritance")

    unimplemented_methods = ['layernorm_fwd', 'gelu', 'silu', 'dropout_fwd']
    for method_name in unimplemented_methods:
        method = getattr(flagos, method_name, None)
        if method is not None:
            method_class = type(flagos)
            if method_name in dir(method_class):
                for cls in method_class.__mro__:
                    if method_name in cls.__dict__:
                        print(f"  {method_name}: defined in {cls.__name__}")
                        break
            else:
                print(f"  {method_name}: inherited")
        else:
            print(f"  {method_name}: NOT FOUND")

    print("\n[Test 5] Checking flagos.py file size")
    flagos_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "transformer_engine_fl", "backends", "flagos", "flagos.py"
    )
    if os.path.exists(flagos_file):
        with open(flagos_file, 'r') as f:
            lines = len(f.readlines())
        print(f"  flagos.py has {lines} lines (reduced from ~1000+ lines)")
        if lines < 400:
            print(f"  [OK] File size significantly reduced!")
        else:
            print(f"  [INFO] File still has {lines} lines")

    print("\n" + "=" * 60)
    print("Fallback Mechanism Test Complete")
    print("=" * 60 + "\n")


def test_actual_fallback_with_tensor():
    print("\n" + "=" * 60)
    print("Testing Actual Fallback with Tensors")
    print("=" * 60 + "\n")

    from transformer_engine.plugins.transformer_engine_fl import get_backend

    os.environ["TE_FL_DEBUG"] = "1"

    flagos = get_backend("flagos")

    print("[Test] rmsnorm_fwd with real tensors (should work directly):")
    try:
        x = torch.randn(2, 4, 8, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        weight = torch.ones(8, dtype=torch.float32, device=x.device)

        result = flagos.rmsnorm_fwd(
            input=x,
            weight=weight,
            eps=1e-5,
            ln_out=None,
            quantizer=None,
            otype=torch.float32,
            sm_margin=0,
            zero_centered_gamma=False,
        )
        print(f"  [OK] rmsnorm_fwd returned: {type(result)}")
        if isinstance(result, tuple):
            print(f"       Output shapes: {[r.shape if hasattr(r, 'shape') else type(r) for r in result]}")
    except Exception as e:
        print(f"  [FAIL] Error: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Tensor Test Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_fallback_mechanism()

    if torch.cuda.is_available():
        test_actual_fallback_with_tensor()
    else:
        print("\n[SKIP] Tensor test skipped - no CUDA available")
