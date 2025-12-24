# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Any, Callable, Optional

DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"

logger = logging.getLogger("transformer_engine_fl.decorators")

from .logger import debug_print_once


def _log_debug(msg: str) -> None:
    if DEBUG:
        print(f"[Backend] {msg}")
    logger.debug(msg)


def debug_print(func_name: str, *args, **kwargs) -> None:
    if DEBUG:
        print(f"[Backend] Calling {func_name}")
        if args:
            print(f"  args: {[type(a).__name__ for a in args[:5]]}...")
        if kwargs:
            print(f"  kwargs: {list(kwargs.keys())[:5]}...")


def _get_operator_registry():
    _op_reg = (
        sys.modules.get("transformer_engine.plugins.transformer_engine_fl.operator_registry") or
        sys.modules.get("transformer_engine_fl.operator_registry")
    )
    if _op_reg is None:
        try:
            from . import operator_registry as _op_reg
        except ImportError:
            _op_reg = None
    return _op_reg


def _get_policy():
    try:
        from .policy import get_policy
        return get_policy()
    except ImportError:
        return None


def _is_strict_mode() -> bool:
    policy = _get_policy()
    if policy is not None:
        return policy.strict
    return os.environ.get("TE_FL_STRICT", "0") == "1"


def with_fallback(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')

        if DEBUG:
            debug_print_once(func_name, backend_name, *args, **kwargs)

        op_reg = _get_operator_registry()
        if op_reg is not None:
            explicit_backend = op_reg.get_operator_backend(func_name)
            priorities = op_reg.get_operator_priority(func_name)

            if explicit_backend or priorities:
                try:
                    resolved_backend, resolved_method = op_reg.resolve_operator_impl(func_name)

                    if resolved_method is not None and resolved_backend != backend_name:
                        if DEBUG:
                            _log_debug(f"[{backend_name}] {func_name} -> delegating to {resolved_backend} (operator priority)")
                        return resolved_method(*args, **kwargs)
                except RuntimeError as e:
                    raise

        try:
            result = func(self, *args, **kwargs)
            return result

        except NotImplementedError as e:
            if _is_strict_mode():
                if DEBUG:
                    _log_debug(f"[{backend_name}] {func_name} not implemented, strict mode enabled - raising")
                raise RuntimeError(
                    f"Operator '{func_name}' not implemented in backend '{backend_name}' "
                    f"and fallback is disabled (strict mode)"
                ) from e

            if DEBUG:
                _log_debug(f"[{backend_name}] {func_name} not implemented, trying fallback...")

            if op_reg is not None:
                try:
                    resolved_backend, resolved_method = op_reg.resolve_operator_impl(func_name)
                    if resolved_method is not None and resolved_backend != backend_name:
                        if DEBUG:
                            _log_debug(f"[{backend_name}] {func_name} -> falling back to {resolved_backend} (operator registry)")
                        return resolved_method(*args, **kwargs)
                except RuntimeError:
                    pass

            if not hasattr(self, '_get_fallback_backend'):
                if DEBUG:
                    _log_debug(f"[{backend_name}] No _get_fallback_backend() method, re-raising")
                raise

            fallback = self._get_fallback_backend()
            if fallback is None:
                if DEBUG:
                    _log_debug(f"[{backend_name}] No fallback backend available, re-raising")
                raise

            fallback_func = getattr(fallback, func_name, None)
            if fallback_func is None:
                if DEBUG:
                    _log_debug(f"[{backend_name}] Fallback backend has no {func_name}, re-raising")
                raise

            if DEBUG:
                fallback_name = getattr(fallback, 'name', 'unknown')
                _log_debug(f"[{backend_name}] Falling back to {fallback_name} for {func_name}")

            return fallback_func(*args, **kwargs)

        except Exception as e:
            if DEBUG:
                _log_debug(f"[{backend_name}] {func_name} failed with error: {e}")
            raise

    return wrapper


def with_operator_priority(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')

        op_reg = _get_operator_registry()
        if op_reg is not None:
            explicit_backend = op_reg.get_operator_backend(func_name)
            priorities = op_reg.get_operator_priority(func_name)

            if explicit_backend or priorities:
                try:
                    resolved_backend, resolved_method = op_reg.resolve_operator_impl(func_name)

                    if resolved_method is not None and resolved_backend != backend_name:
                        if DEBUG:
                            _log_debug(f"[{backend_name}] {func_name} -> delegating to {resolved_backend}")
                        return resolved_method(*args, **kwargs)
                except RuntimeError:
                    raise

        return func(self, *args, **kwargs)

    return wrapper


def with_debug(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')

        if DEBUG:
            debug_print_once(func_name, backend_name, *args, **kwargs)

        try:
            result = func(self, *args, **kwargs)
            return result
        except Exception as e:
            if DEBUG:
                _log_debug(f"[{backend_name}] {func_name} failed with error: {e}")
            raise

    return wrapper


def with_strict_check(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')

        try:
            return func(self, *args, **kwargs)
        except NotImplementedError as e:
            if _is_strict_mode():
                raise RuntimeError(
                    f"Operator '{func_name}' not implemented in backend '{backend_name}' "
                    f"(strict mode enabled)"
                ) from e
            raise

    return wrapper


def with_policy_check(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')
        vendor = getattr(self, 'vendor', None)

        policy = _get_policy()
        if policy is not None and vendor is not None:
            if not policy.is_vendor_allowed(vendor):
                raise RuntimeError(
                    f"Backend '{backend_name}' (vendor: {vendor}) is denied by policy "
                    f"for operator '{func_name}'"
                )

        return func(self, *args, **kwargs)

    return wrapper


def with_full_dispatch(func: Callable) -> Callable:
    return with_fallback(func)


def create_dispatch_decorator(
    enable_fallback: bool = True,
    enable_operator_priority: bool = True,
    enable_strict_check: bool = True,
    enable_debug: bool = True,
) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        result = func

        if enable_debug:
            result = with_debug(result)

        if enable_strict_check and not enable_fallback:
            result = with_strict_check(result)

        if enable_operator_priority and not enable_fallback:
            result = with_operator_priority(result)

        if enable_fallback:
            result = with_fallback(result)

        return result

    return decorator
