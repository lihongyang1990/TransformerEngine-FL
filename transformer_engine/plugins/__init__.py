# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Transformer Engine Plugins

This package provides the unified backend system for TransformerEngine.

Usage:
    # Method 1: Use tex_interface directly
    from transformer_engine.plugins.tex_interface import get_current_backend
    tex = get_current_backend()
    tex.rmsnorm_fwd(...)

    # Method 2: Import the drop-in replacement module
    from transformer_engine.plugins import tex
    tex.rmsnorm_fwd(...)

    # Method 3: For seamless replacement, in your code replace:
    #   import transformer_engine_torch as tex
    # with:
    #   from transformer_engine.plugins.transformer_engine_fl_torch import *
    #   # Then use as if you had: import transformer_engine_fl_torch as tex
"""

# Re-export the tex_interface for convenience
from .tex_interface import (
    TEXBackendBase,
    TEXModule,
    register_backend,
    get_backend,
    get_current_backend,
    set_backend,
    list_backends,
)

# Create a convenient 'tex' alias that works like transformer_engine_torch
from .tex_interface.registry import get_tex_module as _get_tex_module

def __getattr__(name):
    """Lazy loading for tex module."""
    if name == "tex":
        return _get_tex_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TEXBackendBase",
    "TEXModule",
    "register_backend",
    "get_backend",
    "get_current_backend",
    "set_backend",
    "list_backends",
    "tex",  # Lazy-loaded tex module
]
