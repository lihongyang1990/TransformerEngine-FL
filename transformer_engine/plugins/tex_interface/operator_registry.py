# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Operator-Level Priority Registry for TEX Interface

This module provides fine-grained control over which backend implementation
is used for each operator. Instead of selecting a single backend for all
operations, users can specify different backends for different operators.

Example usage:
    # Set rmsnorm to use flaggems backend, but gelu to use torch backend
    set_operator_backend("rmsnorm_fwd", "flaggems")
    set_operator_backend("gelu", "torch")

    # Or set priorities per operator
    set_operator_priority("rmsnorm_fwd", {"flaggems": 200, "torch": 100})
    set_operator_priority("gelu", {"torch": 200, "flaggems": 100})  # torch first for gelu

    # Environment variable support:
    # TE_OP_BACKEND_rmsnorm_fwd=flaggems
    # TE_OP_BACKEND_gelu=torch
    # TE_OP_PRIORITY_rmsnorm_fwd=flaggems:200,torch:100
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
from functools import lru_cache

from .base import TEXBackendBase


# =============================================================================
# Global Registry for Operator-Level Configuration
# =============================================================================

# Operator -> Backend name mapping (explicit backend selection)
_OPERATOR_BACKEND_MAP: Dict[str, str] = {}

# Operator -> {backend_name: priority} mapping (priority-based selection)
_OPERATOR_PRIORITY_MAP: Dict[str, Dict[str, int]] = {}

# Cache for resolved operator implementations
_OPERATOR_IMPL_CACHE: Dict[str, Tuple[str, Callable]] = {}

# Debug mode
_DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"


# =============================================================================
# Configuration API
# =============================================================================

def set_operator_backend(op_name: str, backend_name: str) -> None:
    """
    Set a specific backend for an operator.

    This takes precedence over priority-based selection.

    Args:
        op_name: Operator name (e.g., "rmsnorm_fwd", "gelu", "generic_gemm")
        backend_name: Backend to use (e.g., "flaggems", "torch", "nvidia")

    Example:
        set_operator_backend("rmsnorm_fwd", "flaggems")
        set_operator_backend("gelu", "torch")
    """
    global _OPERATOR_BACKEND_MAP, _OPERATOR_IMPL_CACHE
    _OPERATOR_BACKEND_MAP[op_name] = backend_name
    # Clear cache for this operator
    if op_name in _OPERATOR_IMPL_CACHE:
        del _OPERATOR_IMPL_CACHE[op_name]
    if _DEBUG:
        print(f"[OpRegistry] Set {op_name} -> {backend_name}")


def set_operator_priority(op_name: str, priorities: Dict[str, int]) -> None:
    """
    Set priority scores for different backends for an operator.

    Higher priority = preferred. The backend with highest priority
    that is available will be used.

    Args:
        op_name: Operator name
        priorities: Dict mapping backend_name -> priority score

    Example:
        set_operator_priority("rmsnorm_fwd", {
            "flaggems": 200,  # Highest priority
            "torch": 100,
            "nvidia": 150,
        })
    """
    global _OPERATOR_PRIORITY_MAP, _OPERATOR_IMPL_CACHE
    _OPERATOR_PRIORITY_MAP[op_name] = priorities.copy()
    # Clear cache for this operator
    if op_name in _OPERATOR_IMPL_CACHE:
        del _OPERATOR_IMPL_CACHE[op_name]
    if _DEBUG:
        print(f"[OpRegistry] Set priorities for {op_name}: {priorities}")


def get_operator_backend(op_name: str) -> Optional[str]:
    """Get the explicitly configured backend for an operator."""
    # Check environment variable first
    env_key = f"TE_OP_BACKEND_{op_name}"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    return _OPERATOR_BACKEND_MAP.get(op_name)


def get_operator_priority(op_name: str) -> Optional[Dict[str, int]]:
    """Get the priority configuration for an operator."""
    # Check environment variable first
    env_key = f"TE_OP_PRIORITY_{op_name}"
    env_val = os.environ.get(env_key)
    if env_val:
        # Parse format: "flaggems:200,torch:100"
        priorities = {}
        for item in env_val.split(","):
            parts = item.strip().split(":")
            if len(parts) == 2:
                backend, priority = parts
                priorities[backend.strip()] = int(priority.strip())
        if priorities:
            return priorities
    return _OPERATOR_PRIORITY_MAP.get(op_name)


def clear_operator_config(op_name: Optional[str] = None) -> None:
    """
    Clear operator configuration.

    Args:
        op_name: If provided, clear only this operator. Otherwise clear all.
    """
    global _OPERATOR_BACKEND_MAP, _OPERATOR_PRIORITY_MAP, _OPERATOR_IMPL_CACHE

    if op_name:
        _OPERATOR_BACKEND_MAP.pop(op_name, None)
        _OPERATOR_PRIORITY_MAP.pop(op_name, None)
        _OPERATOR_IMPL_CACHE.pop(op_name, None)
    else:
        _OPERATOR_BACKEND_MAP.clear()
        _OPERATOR_PRIORITY_MAP.clear()
        _OPERATOR_IMPL_CACHE.clear()


def list_operator_config() -> Dict[str, Any]:
    """List all operator configurations."""
    return {
        "explicit_backends": _OPERATOR_BACKEND_MAP.copy(),
        "priority_configs": _OPERATOR_PRIORITY_MAP.copy(),
    }


# =============================================================================
# Batch Configuration
# =============================================================================

def configure_operators(config: Dict[str, Any]) -> None:
    """
    Configure multiple operators at once.

    Args:
        config: Dict with operator names as keys and either:
            - str: backend name (explicit selection)
            - dict: priority mapping

    Example:
        configure_operators({
            "rmsnorm_fwd": "flaggems",  # Explicit backend
            "rmsnorm_bwd": "flaggems",
            "gelu": {"torch": 200, "flaggems": 100},  # Priority-based
            "generic_gemm": {"flaggems": 200, "torch": 50},
        })
    """
    for op_name, value in config.items():
        if isinstance(value, str):
            set_operator_backend(op_name, value)
        elif isinstance(value, dict):
            set_operator_priority(op_name, value)
        else:
            raise ValueError(f"Invalid config for {op_name}: {value}")


def configure_from_env() -> None:
    """
    Load operator configuration from environment variables.

    Supports:
        TE_OP_BACKEND_<op_name>=<backend_name>
        TE_OP_PRIORITY_<op_name>=<backend1>:<priority1>,<backend2>:<priority2>

    Example:
        TE_OP_BACKEND_rmsnorm_fwd=flaggems
        TE_OP_PRIORITY_gelu=torch:200,flaggems:100
    """
    for key, value in os.environ.items():
        if key.startswith("TE_OP_BACKEND_"):
            op_name = key[len("TE_OP_BACKEND_"):]
            set_operator_backend(op_name, value)
        elif key.startswith("TE_OP_PRIORITY_"):
            op_name = key[len("TE_OP_PRIORITY_"):]
            # Parse format: "flaggems:200,torch:100"
            priorities = {}
            for item in value.split(","):
                parts = item.strip().split(":")
                if len(parts) == 2:
                    backend, priority = parts
                    priorities[backend.strip()] = int(priority.strip())
            if priorities:
                set_operator_priority(op_name, priorities)


# =============================================================================
# Operator Resolution
# =============================================================================

def get_available_backends() -> Dict[str, TEXBackendBase]:
    """Get all available backend instances."""
    # Import registry here to avoid circular imports
    _reg = (
        sys.modules.get("transformer_engine.plugins.tex_interface.registry") or
        sys.modules.get("tex_interface.registry")
    )
    if _reg is None:
        from . import registry as _reg

    backends = {}
    for backend_info in _reg.list_backends():
        name = backend_info["name"]
        if backend_info.get("available", False):
            try:
                backends[name] = _reg._get_backend_instance(name)
            except Exception:
                pass
    return backends


def resolve_operator_impl(
    op_name: str,
    available_backends: Optional[Dict[str, TEXBackendBase]] = None,
) -> Tuple[Optional[str], Optional[Callable]]:
    """
    Resolve which backend implementation to use for an operator.

    Resolution order:
    1. Explicit backend setting (set_operator_backend or TE_OP_BACKEND_*)
    2. Priority-based selection (set_operator_priority or TE_OP_PRIORITY_*)
    3. Default to backend-level priority (existing behavior)

    Args:
        op_name: Name of the operator
        available_backends: Optional dict of available backends

    Returns:
        Tuple of (backend_name, method) or (None, None) if not found
    """
    # Check cache first
    if op_name in _OPERATOR_IMPL_CACHE:
        return _OPERATOR_IMPL_CACHE[op_name]

    if available_backends is None:
        available_backends = get_available_backends()

    if not available_backends:
        return None, None

    # 1. Check explicit backend setting
    explicit_backend = get_operator_backend(op_name)
    if explicit_backend:
        if explicit_backend in available_backends:
            backend = available_backends[explicit_backend]
            method = getattr(backend, op_name, None)
            if method is not None:
                result = (explicit_backend, method)
                _OPERATOR_IMPL_CACHE[op_name] = result
                if _DEBUG:
                    print(f"[OpRegistry] {op_name} -> {explicit_backend} (explicit)")
                return result
        if _DEBUG:
            print(f"[OpRegistry] Warning: explicit backend {explicit_backend} not available for {op_name}")

    # 2. Check priority-based selection
    priorities = get_operator_priority(op_name)
    if priorities:
        # Sort by priority (highest first)
        sorted_backends = sorted(
            [(name, prio) for name, prio in priorities.items() if name in available_backends],
            key=lambda x: x[1],
            reverse=True
        )
        for backend_name, priority in sorted_backends:
            backend = available_backends[backend_name]
            method = getattr(backend, op_name, None)
            if method is not None:
                # Check if method is actually implemented (not just raises NotImplementedError)
                # We can't easily check this without calling, so we trust the registration
                result = (backend_name, method)
                _OPERATOR_IMPL_CACHE[op_name] = result
                if _DEBUG:
                    print(f"[OpRegistry] {op_name} -> {backend_name} (priority={priority})")
                return result

    # 3. Fall back to backend-level priority
    # Sort backends by their default priority
    sorted_backends = sorted(
        available_backends.items(),
        key=lambda x: x[1].priority,
        reverse=True
    )
    for backend_name, backend in sorted_backends:
        method = getattr(backend, op_name, None)
        if method is not None:
            result = (backend_name, method)
            _OPERATOR_IMPL_CACHE[op_name] = result
            if _DEBUG:
                print(f"[OpRegistry] {op_name} -> {backend_name} (default priority={backend.priority})")
            return result

    return None, None


def call_operator(
    op_name: str,
    *args,
    available_backends: Optional[Dict[str, TEXBackendBase]] = None,
    **kwargs
) -> Any:
    """
    Call an operator using the best available implementation.

    This function resolves the best backend for the operator and calls it.

    Args:
        op_name: Name of the operator
        *args, **kwargs: Arguments to pass to the operator
        available_backends: Optional dict of available backends

    Returns:
        Result from the operator

    Raises:
        NotImplementedError: If no implementation is available
    """
    backend_name, method = resolve_operator_impl(op_name, available_backends)

    if method is None:
        raise NotImplementedError(
            f"No implementation found for operator '{op_name}'. "
            f"Available backends: {list(get_available_backends().keys())}"
        )

    return method(*args, **kwargs)


# =============================================================================
# Decorator for Operator-Level Priority
# =============================================================================

def with_operator_priority(func: Callable) -> Callable:
    """
    Decorator that enables operator-level priority selection.

    This decorator wraps a backend method to:
    1. Check if there's an operator-level override
    2. If yes, delegate to the specified backend
    3. If no, execute the original method with fallback support

    Usage:
        class MyBackend(TEXBackendBase):
            @with_operator_priority
            def rmsnorm_fwd(self, ...):
                # Implementation
                pass
    """
    import functools

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__

        # Check if there's an operator-level override
        explicit_backend = get_operator_backend(func_name)
        priorities = get_operator_priority(func_name)

        if explicit_backend or priorities:
            # Use operator-level resolution
            backend_name, method = resolve_operator_impl(func_name)

            if method is not None and backend_name != getattr(self, 'name', None):
                # Delegate to the resolved backend
                if _DEBUG:
                    print(f"[OpPriority] {func_name}: delegating from {self.name} to {backend_name}")
                return method(*args, **kwargs)

        # Execute the original method
        return func(self, *args, **kwargs)

    return wrapper


# =============================================================================
# Initialize from environment on module load
# =============================================================================

# Auto-load configuration from environment variables
configure_from_env()
