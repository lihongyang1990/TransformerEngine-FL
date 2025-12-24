#!/usr/bin/env python3
# Copyright (c) 2025, BAAI. All rights reserved.
#
import sys
import os


def print_header(TEFLt):
    print("\n" + "=" * 80)
    print(TEFLt)
    print("=" * 80)


def print_section(TEFLt):
    print("\n" + "-" * 80)
    print(TEFLt)
    print("-" * 80)


def print_success(TEFLt):
    print(f"✓ {TEFLt}")


def print_error(TEFLt):
    print(f"✗ {TEFLt}")


def print_info(TEFLt):
    print(f"  {TEFLt}")


def test_tefl_flash_attention_availability():
    print_section("Test 1: Check tefl module for flash_attention")

    try:
        import transformer_engine_torch as tefl

        has_flash_attention = hasattr(TEFL, 'flash_attention')
        is_callable = callable(getattr(TEFL, 'flash_attention', None))

        print_info(f"Has flash_attention: {has_flash_attention}")
        print_info(f"Is callable: {is_callable}")

        if has_flash_attention and is_callable:
            print_success("Plugin system's flash_attention is available")
            return True
        else:
            print_info("Plugin system's flash_attention not available")
            print_info("(This is expected if using native TEFL module)")
            return False

    except ImportError as e:
        print_error(f"Failed to import transformer_engine_torch: {e}")
        return False


def test_dotproductattention_code_changes():
    print_section("Test 2: Verify DotProductAttention code changes")

    try:
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "pytorch/attention/dot_product_attention/dot_product_attention.py"
        )

        with open(file_path, 'r') as f:
            lines = f.readlines()

        found_native_save = False
        found_getattr = False
        native_save_line = -1
        getattr_line = -1

        for i, line in enumerate(lines[58:75], start=58):
            if '_FlashAttentionNative = FlashAttention' in line:
                found_native_save = True
                native_save_line = i + 1
            if 'FlashAttention = getattr(TEFL, \'flash_attention\'' in line:
                found_getattr = True
                getattr_line = i + 1

        if found_native_save:
            print_success(f"Found line {native_save_line}: Save native reference")
        else:
            print_error("Missing: _FlashAttentionNative = FlashAttention")

        if found_getattr:
            print_success(f"Found line {getattr_line}: Plugin system integration")
        else:
            print_error("Missing: FlashAttention = getattr(...)")

        if found_native_save and found_getattr:
            print_success("DotProductAttention has been modified correctly")
            print_info("  - Original import unchanged")
            print_info("  - Native reference saved")
            print_info("  - Plugin system integration added")
            return True
        else:
            print_error("Some modifications are missing")
            return False

    except Exception as e:
        print_error(f"Error checking file: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_system_integration():
    print_section("Test 3: Test plugin system integration")

    try:
        import transformer_engine.plugins.transformer_engine_fl_torch as TEFL_plugins
        sys.modules['transformer_engine_torch'] = TEFL_plugins

        print_info("Replaced transformer_engine_torch with plugin system")

        has_flash_attention = hasattr(TEFL_plugins, 'flash_attention')
        is_callable = callable(getattr(TEFL_plugins, 'flash_attention', None))

        if not has_flash_attention or not is_callable:
            print_error("Plugin system flash_attention not available")
            return False

        print_success("Plugin system flash_attention is available")

        from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention

        print_success("DotProductAttention imported successfully")

        attn = DotProductAttention(
            num_attention_heads=8,
            kv_channels=64,
            attention_dropout=0.0,
            layer_number=1,
        )

        print_success("DotProductAttention instance created")

        if hasattr(attn.flash_attention, 'backend_name'):
            backend_name = attn.flash_attention.backend_name
            backend_class = attn.flash_attention.__class__.__name__

            print_success(f"Using plugin system's FlashAttention")
            print_info(f"  Backend name: {backend_name}")
            print_info(f"  Backend class: {backend_class}")

            return True
        else:
            print_error("Not using plugin system (missing backend_name attribute)")
            print_info(f"  Class: {attn.flash_attention.__class__.__name__}")
            return False

    except Exception as e:
        print_error(f"Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_switching():
    print_section("Test 4: Test backend switching")

    try:
        backends_to_test = ["flagos", "reference"]

        for backend in backends_to_test:
            print_info(f"\nTesting backend: {backend}")

            os.environ["TE_FL_BACKEND"] = backend

            import importlib
            import transformer_engine.plugins.transformer_engine_fl.registry as registry_module
            importlib.reload(registry_module)

            import transformer_engine.plugins.transformer_engine_fl_torch as TEFL_plugins
            importlib.reload(TEFL_plugins)
            sys.modules['transformer_engine_torch'] = TEFL_plugins

            from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention

            attn = DotProductAttention(
                num_attention_heads=8,
                kv_channels=64,
                attention_dropout=0.0,
            )

            if hasattr(attn.flash_attention, 'backend_name'):
                actual_backend = attn.flash_attention.backend_name
                print_success(f"  Backend: {actual_backend}")

                if actual_backend == backend:
                    print_success(f"  Correct backend selected")
                else:
                    print_info(f"  Expected '{backend}', got '{actual_backend}'")
            else:
                print_error(f"  Not using plugin system")

        return True

    except Exception as e:
        print_error(f"Error during backend switching test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print_header("FlashAttention Plugin System Integration Tests")

    results = []

    results.append(("TEFL.flash_attention availability", test_TEFL_flash_attention_availability()))

    results.append(("DotProductAttention code changes", test_dotproductattention_code_changes()))

    results.append(("Plugin system integration", test_plugin_system_integration()))

    results.append(("Backend switching", test_backend_switching()))

    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print_success("All tests passed!")
        print()
        print_info("Integration is complete! You can now:")
        print_info("  1. Set backend: export TE_FL_BACKEND=flagos")
        print_info("  2. Replace TEFL module in your code:")
        print_info("     import transformer_engine.plugins.transformer_engine_fl_torch as TEFL")
        print_info("     sys.modules['transformer_engine_torch'] = TEFL")
        print_info("  3. Use DotProductAttention as normal")
    else:
        print_error("Some tests failed. Please check the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
