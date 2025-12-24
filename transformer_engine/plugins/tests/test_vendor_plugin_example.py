# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Example Vendor Plugin for TE_FL Interface.

This file demonstrates how to create an out-of-tree vendor plugin with MINIMAL code.
Vendors only need to implement:
1. Four required properties/methods: name, vendor, priority, is_available
2. The specific operators they want to optimize

Unimplemented operators will:
- Automatically fallback to other backends via @with_fallback decorator
- Or raise NotImplementedError if no fallback is available

Usage:
    # Test with environment variable:
    TE_FL_PLUGIN_MODULES=test_vendor_plugin_example python your_app.py

    # Or install as a package with entry points:
    # pyproject.toml:
    # [project.entry-points."te_fl.plugins"]
    # example_vendor = "test_vendor_plugin_example:te_fl_register"
"""

from __future__ import annotations

import os
import sys

from transformer_engine.plugins.transformer_engine_fl import TEFLBackendBase, register_backend
from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind
from transformer_engine.plugins.transformer_engine_fl.decorators import with_fallback


# =============================================================================
# Example Vendor Backend - MINIMAL Implementation
# =============================================================================

class ExampleVendorBackend(TEFLBackendBase):
    """
    Example vendor backend implementation.

    This demonstrates the MINIMAL implementation required for a vendor plugin.
    Only 4 core properties/methods are required. All other methods are optional.

    Vendors only implement the operators they have optimized versions for.
    Everything else automatically falls back to other backends.
    """

    NAME = "example_vendor"
    PRIORITY = 160  # Higher than default backends

    def __init__(self):
        """Initialize the backend."""
        self._fallback_backend = None

    # =========================================================================
    # REQUIRED: These 4 are the ONLY abstract methods that MUST be implemented
    # =========================================================================

    @property
    def name(self) -> str:
        """Backend identifier (REQUIRED)."""
        return "example_vendor"

    @property
    def vendor(self) -> str:
        """Vendor display name (REQUIRED)."""
        return "Example Vendor Inc."

    @property
    def priority(self) -> int:
        """Selection priority - higher = preferred (REQUIRED)."""
        return 160

    def is_available(self) -> bool:
        """
        Check if this backend is available (REQUIRED).

        Real vendors would check:
        - Hardware availability (GPU, accelerator)
        - Driver version
        - Required libraries
        """
        # For demo, always available
        return True

    # =========================================================================
    # OPTIONAL: Implementation type (for policy-based selection)
    # =========================================================================

    @property
    def impl_kind(self) -> BackendImplKind:
        """This is a VENDOR implementation."""
        return BackendImplKind.VENDOR

    @classmethod
    def check_available(cls) -> bool:
        """Class-level availability check (without instantiation)."""
        return True

    # =========================================================================
    # OPTIONAL: Helper for fallback
    # =========================================================================

    def _get_fallback_backend(self):
        """Get fallback backend for unimplemented operations."""
        if self._fallback_backend is None:
            from transformer_engine.plugins.transformer_engine_fl import get_backend
            try:
                self._fallback_backend = get_backend("reference")
            except Exception:
                pass
        return self._fallback_backend

    # =========================================================================
    # OPTIONAL: Implement ONLY the operators you want to optimize
    # All others automatically fallback or raise NotImplementedError
    # =========================================================================

    @with_fallback
    def rmsnorm_fwd(self, x, weight, eps=1e-6, **kwargs):
        """
        Example vendor-optimized RMSNorm forward.

        Real implementation would use vendor-specific kernels.
        Here we demonstrate fallback behavior.
        """
        print(f"[{self.name}] Running optimized rmsnorm_fwd")
        # This will trigger fallback to torch backend
        raise NotImplementedError("Demo - fallback to reference")

    @with_fallback
    def gelu(self, x, quantizer=None):
        """Example GELU implementation - optimized for this vendor."""
        print(f"[{self.name}] Running optimized gelu")
        # Real implementation here...
        raise NotImplementedError("Demo - fallback to reference")

    # =========================================================================
    # That's it! No need to implement 100+ other methods.
    # They all have default implementations in TEFLBackendBase that raise
    # NotImplementedError, which @with_fallback will catch and redirect.
    # =========================================================================


# =============================================================================
# Plugin Registration Function
# =============================================================================

def te_fl_register(registry_module):
    """
    Entry point for plugin registration.

    This function is called automatically when the plugin is discovered
    via entry points or TE_FL_PLUGIN_MODULES.

    Args:
        registry_module: The registry module (interface.registry)
    """
    print("[ExampleVendorPlugin] Plugin registered successfully!")


# Also support the alternative function name
register = te_fl_register


# =============================================================================
# Manual Registration (when imported directly)
# =============================================================================

def _register_if_not_registered():
    """Register backend if not already registered."""
    try:
        from transformer_engine.plugins.transformer_engine_fl import is_backend_registered, register_backend

        if not is_backend_registered("example_vendor"):
            register_backend(ExampleVendorBackend)
            print("[ExampleVendorPlugin] Backend registered via direct import")
    except Exception as e:
        print(f"[ExampleVendorPlugin] Registration failed: {e}")


# Auto-register on import
_register_if_not_registered()


# =============================================================================
# Test
# =============================================================================

def test_example_vendor():
    """Test the example vendor backend."""
    from transformer_engine.plugins.transformer_engine_fl import list_backends, get_backend

    print("\n=== Testing Example Vendor Plugin ===\n")

    # List all backends
    print("Available backends:")
    for backend in list_backends():
        status = "CURRENT" if backend.get("current") else ("available" if backend.get("available") else "unavailable")
        print(f"  - {backend['name']}: priority={backend['priority']}, {status}")

    # Check if our backend is registered
    backend_names = [b["name"] for b in list_backends()]
    if "example_vendor" in backend_names:
        print("\n[OK] Example vendor backend is registered!")

        # Try to use it
        try:
            backend = get_backend("example_vendor")
            print(f"[OK] Got backend: {backend.name} (vendor: {backend.vendor})")
            print(f"     impl_kind: {backend.impl_kind}")
            print(f"     priority: {backend.priority}")
            print(f"     available: {backend.is_available()}")

            # Test gelu with fallback
            print("\n[Test] Calling gelu (should trigger fallback):")
            try:
                import torch
                x = torch.randn(2, 3, 4)
                result = backend.gelu(x, quantizer=None)
                print(f"     [OK] GELU executed successfully with fallback")
                print(f"     Result shape: {result.shape}")
            except Exception as e:
                print(f"     [ERROR] GELU failed: {e}")

        except Exception as e:
            print(f"[ERROR] Failed to get backend: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[WARNING] Example vendor backend not found in registry")

    print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    test_example_vendor()
