# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Transformer Engine Torch Interface (tex_interface)

This module defines the complete interface that matches transformer_engine_torch (tex).
All vendors should implement this interface, and the usage should be:

    # Original:
    import transformer_engine_torch as tex

    # With vendor plugin (e.g., for Hygon):
    import transformer_engine_fl_torch as tex

    # Then use tex.* exactly the same way

Architecture:
    tex_interface/
        __init__.py           # This file - exports the base module class
        base.py               # Abstract base class with all tex interfaces
        registry.py           # Backend registry and selection
        operator_registry.py  # Operator-level priority configuration
        nvidia_backend.py     # NVIDIA backend (wraps original transformer_engine_torch)
        template_backend.py   # Template for vendors to implement

Usage for vendors:
    1. Inherit from TEXBackendBase in base.py
    2. Implement all required methods
    3. Register your backend
    4. Build as transformer_engine_fl_torch

Operator-Level Priority:
    You can configure which backend to use for each operator:

    # Set specific backend for an operator
    set_operator_backend("rmsnorm_fwd", "flaggems")
    set_operator_backend("gelu", "torch")

    # Or set priorities per operator (higher = preferred)
    set_operator_priority("rmsnorm_fwd", {"flaggems": 200, "torch": 100})

    # Environment variables also work:
    # TE_OP_BACKEND_rmsnorm_fwd=flaggems
    # TE_OP_PRIORITY_gelu=torch:200,flaggems:100
"""

from .base import TEXBackendBase, TEXModule
from .registry import (
    register_backend,
    get_backend,
    get_current_backend,
    set_backend,
    list_backends,
)
from .decorators import with_fallback, with_debug, with_operator_priority

# Operator-level priority configuration
from .operator_registry import (
    set_operator_backend,
    set_operator_priority,
    get_operator_backend,
    get_operator_priority,
    clear_operator_config,
    list_operator_config,
    configure_operators,
    configure_from_env,
    resolve_operator_impl,
    call_operator,
)

# IMPORTANT: Import backends to trigger registration via @register_backend decorator
from . import backends

__all__ = [
    # Base classes
    "TEXBackendBase",
    "TEXModule",
    # Backend registry
    "register_backend",
    "get_backend",
    "get_current_backend",
    "set_backend",
    "list_backends",
    # Decorators
    "with_fallback",
    "with_debug",
    "with_operator_priority",
    # Operator-level priority (NEW)
    "set_operator_backend",
    "set_operator_priority",
    "get_operator_backend",
    "get_operator_priority",
    "clear_operator_config",
    "list_operator_config",
    "configure_operators",
    "configure_from_env",
    "resolve_operator_impl",
    "call_operator",
]
