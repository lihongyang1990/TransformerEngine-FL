# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Common decorators for TEX Interface backends.

This module provides reusable decorators that can be used by all backend implementations.
"""

import os
import sys
import functools
from typing import Callable, Any, Optional, Dict


# Debug mode: set TE_FL_DEBUG=1 to enable debug printing
DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"

# Import logger utilities
from .logger import debug_print_once


def debug_print(func_name: str, *args, **kwargs):
    """Print debug information for function calls."""
    if DEBUG:
        print(f"[Backend] Calling {func_name}")
        if args:
            print(f"  args: {[type(a).__name__ for a in args[:5]]}...")
        if kwargs:
            print(f"  kwargs: {list(kwargs.keys())[:5]}...")


def _get_operator_registry():
    """Get the operator registry module (lazy import to avoid circular deps)."""
    _op_reg = (
        sys.modules.get("transformer_engine.plugins.tex_interface.operator_registry") or
        sys.modules.get("tex_interface.operator_registry")
    )
    if _op_reg is None:
        try:
            from . import operator_registry as _op_reg
        except ImportError:
            _op_reg = None
    return _op_reg


def with_fallback(func: Callable) -> Callable:
    """
    Decorator to automatically fallback to another backend on NotImplementedError.

    This decorator wraps backend methods to provide:
    1. Operator-level priority selection (NEW)
    2. Automatic fallback functionality
    3. Optional debug printing when DEBUG mode is enabled

    The resolution order is:
    1. Check operator-level configuration (explicit backend or priorities)
    2. If configured, delegate to the best available implementation
    3. Otherwise, try the current backend
    4. On NotImplementedError, fallback to another backend

    Usage:
        class MyBackend(TEXBackendBase):
            @with_fallback
            def some_method(self, ...):
                raise NotImplementedError("Not implemented yet")

    Requirements:
        - The backend class must implement `_get_fallback_backend()` method
        - The fallback backend should have the same method available

    Args:
        func: The backend method to wrap

    Returns:
        Wrapped function with operator priority and fallback capability
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')

        # Debug printing (optional feature) - print only once per function
        if DEBUG:
            debug_print_once(func_name, backend_name, *args, **kwargs)

        # =====================================================================
        # Step 1: Check operator-level configuration
        # =====================================================================
        op_reg = _get_operator_registry()
        if op_reg is not None:
            explicit_backend = op_reg.get_operator_backend(func_name)
            priorities = op_reg.get_operator_priority(func_name)

            if explicit_backend or priorities:
                # Resolve the best backend for this operator
                resolved_backend, resolved_method = op_reg.resolve_operator_impl(func_name)

                if resolved_method is not None and resolved_backend != backend_name:
                    # Delegate to the resolved backend
                    if DEBUG:
                        print(f"[{backend_name}] {func_name} -> delegating to {resolved_backend} (operator priority)")
                    return resolved_method(*args, **kwargs)

        # =====================================================================
        # Step 2: Try the current backend
        # =====================================================================
        try:
            result = func(self, *args, **kwargs)
            return result

        except NotImplementedError as e:
            # =====================================================================
            # Step 3: Fallback to another backend
            # =====================================================================
            if DEBUG:
                print(f"[{backend_name}] {func_name} not implemented, trying fallback...")

            # Try operator-level resolution first (if not already tried)
            if op_reg is not None:
                resolved_backend, resolved_method = op_reg.resolve_operator_impl(func_name)
                if resolved_method is not None and resolved_backend != backend_name:
                    if DEBUG:
                        print(f"[{backend_name}] {func_name} -> falling back to {resolved_backend} (operator registry)")
                    return resolved_method(*args, **kwargs)

            # Fall back to the configured fallback backend
            if not hasattr(self, '_get_fallback_backend'):
                if DEBUG:
                    print(f"[{backend_name}] No _get_fallback_backend() method, re-raising")
                raise

            fallback = self._get_fallback_backend()
            if fallback is None:
                if DEBUG:
                    print(f"[{backend_name}] No fallback backend available, re-raising")
                raise

            fallback_func = getattr(fallback, func_name, None)
            if fallback_func is None:
                if DEBUG:
                    print(f"[{backend_name}] Fallback backend has no {func_name}, re-raising")
                raise

            if DEBUG:
                fallback_name = getattr(fallback, 'name', 'unknown')
                print(f"[{backend_name}] Falling back to {fallback_name} for {func_name}")

            return fallback_func(*args, **kwargs)

        except Exception as e:
            if DEBUG:
                print(f"[{backend_name}] {func_name} failed with error: {e}")
            raise

    return wrapper


def with_operator_priority(func: Callable) -> Callable:
    """
    Decorator that ONLY enables operator-level priority selection.

    Unlike with_fallback, this decorator does NOT provide fallback functionality.
    It only checks operator-level configuration and delegates if needed.

    Use this for backends that don't support fallback but want operator-level
    priority support.

    Usage:
        class MyBackend(TEXBackendBase):
            @with_operator_priority
            def rmsnorm_fwd(self, ...):
                # Implementation that must succeed
                pass
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')

        # Check operator-level configuration
        op_reg = _get_operator_registry()
        if op_reg is not None:
            explicit_backend = op_reg.get_operator_backend(func_name)
            priorities = op_reg.get_operator_priority(func_name)

            if explicit_backend or priorities:
                resolved_backend, resolved_method = op_reg.resolve_operator_impl(func_name)

                if resolved_method is not None and resolved_backend != backend_name:
                    if DEBUG:
                        print(f"[{backend_name}] {func_name} -> delegating to {resolved_backend}")
                    return resolved_method(*args, **kwargs)

        # Execute the original method
        return func(self, *args, **kwargs)

    return wrapper


def with_debug(func: Callable) -> Callable:
    """
    Decorator to add debug printing to backend methods.

    This is a lightweight decorator that only provides debug output,
    without any fallback or priority functionality.

    Usage:
        class MyBackend(TEXBackendBase):
            @with_debug
            def some_method(self, ...):
                return result

    Args:
        func: The backend method to wrap

    Returns:
        Wrapped function with debug printing
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')

        # Debug printing - print only once per function
        if DEBUG:
            debug_print_once(func_name, backend_name, *args, **kwargs)

        try:
            result = func(self, *args, **kwargs)
            return result
        except Exception as e:
            if DEBUG:
                print(f"[{backend_name}] {func_name} failed with error: {e}")
            raise

    return wrapper
