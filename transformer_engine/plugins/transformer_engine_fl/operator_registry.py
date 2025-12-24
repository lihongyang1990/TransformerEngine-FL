from __future__ import annotations

import functools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import TEFLBackendBase

from .operator_manager import OperatorManager

_DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"


def _log_debug(msg: str) -> None:
    if _DEBUG:
        print(f"[OpRegistry] {msg}")


def set_operator_backend(op_name: str, backend_name: str) -> None:
    OperatorManager.get_instance().set_operator_backend(op_name, backend_name)


def set_operator_priority(op_name: str, priorities: Dict[str, int]) -> None:
    OperatorManager.get_instance().set_operator_priority(op_name, priorities)


def get_operator_backend(op_name: str) -> Optional[str]:
    return OperatorManager.get_instance().get_operator_backend(op_name)


def get_operator_priority(op_name: str) -> Optional[Dict[str, int]]:
    return OperatorManager.get_instance().get_operator_priority(op_name)


def clear_operator_config(op_name: Optional[str] = None) -> None:
    OperatorManager.get_instance().clear_operator_config(op_name)


def list_operator_config() -> Dict[str, Any]:
    return OperatorManager.get_instance().list_operator_config()


def configure_operators(config: Dict[str, Any]) -> None:
    OperatorManager.get_instance().configure_operators(config)


def configure_from_env() -> None:
    OperatorManager.get_instance().configure_from_env()


def get_available_backends() -> Dict[str, TEFLBackendBase]:
    return OperatorManager.get_instance().get_available_backends()


def resolve_operator_impl(
    op_name: str,
    available_backends: Optional[Dict[str, TEFLBackendBase]] = None,
) -> Tuple[Optional[str], Optional[Callable]]:
    return OperatorManager.get_instance().resolve_operator_impl(op_name, available_backends)


def call_operator(
    op_name: str,
    *args,
    available_backends: Optional[Dict[str, TEFLBackendBase]] = None,
    **kwargs
) -> Any:
    return OperatorManager.get_instance().call_operator(op_name, *args, available_backends, **kwargs)


def with_operator_priority(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        backend_name = getattr(self, 'name', 'Backend')

        explicit_backend = get_operator_backend(func_name)
        priorities = get_operator_priority(func_name)

        if explicit_backend or priorities:
            resolved_backend, method = resolve_operator_impl(func_name)

            if method is not None and resolved_backend != backend_name:
                _log_debug(f"{func_name}: delegating from {backend_name} to {resolved_backend}")
                return method(*args, **kwargs)

        return func(self, *args, **kwargs)

    return wrapper


def clear_cache() -> None:
    OperatorManager.get_instance().clear_cache()


def get_cache_stats() -> Dict[str, Any]:
    return OperatorManager.get_instance().get_cache_stats()


__all__ = [
    "OperatorManager",
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
    "get_available_backends",
    "clear_cache",
    "get_cache_stats",
    "with_operator_priority",
]
