# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Transformer Engine FL Torch Interface

This module serves as a drop-in replacement for transformer_engine_torch.
It provides the same interface but allows using different hardware backends.

Usage:
    # Original code:
    import transformer_engine_torch as tex
    tex.rmsnorm_fwd(...)

    # With this module (just change the import):
    import transformer_engine_fl_torch as tex
    tex.rmsnorm_fwd(...)

Backend Selection:
    By default, the best available backend is auto-selected.
    You can override this with environment variables:

    # Force NVIDIA backend
    export TE_BACKEND=nvidia

    # Force a specific vendor backend
    export TE_BACKEND=hygon

Available Backends:
    - nvidia: NVIDIA CUDA (wraps original transformer_engine_torch)
    - (vendors can add their own backends)
"""

import os
import sys
from typing import Any

# Import the tex interface components
# Support both standalone import and as part of transformer_engine.plugins
import importlib.util
from pathlib import Path

# Determine the path to tex_interface relative to this file
_this_dir = Path(__file__).parent
_tex_interface_dir = _this_dir.parent / "tex_interface"

# Load tex_interface modules directly to avoid triggering main transformer_engine init
def _load_module_from_path(name: str, path: Path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load base module
_base_module = _load_module_from_path(
    "tex_interface.base",
    _tex_interface_dir / "base.py"
)

# Load registry module
_registry_module = _load_module_from_path(
    "tex_interface.registry",
    _tex_interface_dir / "registry.py"
)

# Load backends init to register backends
_backends_init = _tex_interface_dir / "backends" / "__init__.py"
if _backends_init.exists():
    _backends_module = _load_module_from_path(
        "tex_interface.backends",
        _backends_init
    )

# Import what we need from loaded modules
DType = _base_module.DType
FP8FwdTensors = _base_module.FP8FwdTensors
FP8BwdTensors = _base_module.FP8BwdTensors
FP8TensorMeta = _base_module.FP8TensorMeta
TEXBackendBase = _base_module.TEXBackendBase

get_current_backend = _registry_module.get_current_backend
get_backend = _registry_module.get_backend
set_backend = _registry_module.set_backend
list_backends = _registry_module.list_backends


# =============================================================================
# Module-level Interface
# =============================================================================

# Get the current backend
_backend: TEXBackendBase = None


def _get_backend() -> TEXBackendBase:
    """Get or initialize the backend."""
    global _backend
    if _backend is None:
        _backend = get_current_backend()
    return _backend


def _reset_backend():
    """Reset the backend (for testing or switching)."""
    global _backend
    _backend = None


# =============================================================================
# Export all tex interfaces at module level
# =============================================================================

# Enums and data classes
__all__ = [
    # Enums
    "DType",
    "FP8FwdTensors",
    "FP8BwdTensors",
    # Data classes
    "FP8TensorMeta",
    # Backend management
    "get_backend",
    "set_backend",
    "list_backends",
]


# =============================================================================
# Dynamic attribute access for backend methods
# =============================================================================

class _TEXModuleWrapper:
    """
    A module wrapper that dynamically delegates to the current backend.

    This allows:
        import transformer_engine_fl_torch as tex
        tex.rmsnorm_fwd(...)

    To work exactly like:
        import transformer_engine_torch as tex
        tex.rmsnorm_fwd(...)
    """

    def __init__(self, module_name: str):
        self._module_name = module_name
        # Store original module attributes
        self._original_module = sys.modules[module_name]

    def __getattr__(self, name: str) -> Any:
        # First check if it's a module-level attribute
        if name.startswith('_'):
            raise AttributeError(f"module '{self._module_name}' has no attribute '{name}'")

        # Check if it's an exported constant/class
        if name in ('DType', 'FP8FwdTensors', 'FP8BwdTensors', 'FP8TensorMeta'):
            return getattr(self._original_module, name)

        if name in ('get_backend', 'set_backend', 'list_backends'):
            return getattr(self._original_module, name)

        # Otherwise delegate to backend
        backend = _get_backend()
        return getattr(backend, name)

    def __dir__(self):
        # List available attributes from backend plus module-level ones
        backend = _get_backend()
        return list(set(dir(backend) + __all__))


# Replace this module with the wrapper
sys.modules[__name__] = _TEXModuleWrapper(__name__)
