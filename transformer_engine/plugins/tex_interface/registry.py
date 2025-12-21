# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Backend Registry for Transformer Engine Torch Interface

This module provides registration and selection of TEX backends.
Vendors register their backends here, and the system auto-selects
the best available backend.
"""

import os
import sys
from typing import Dict, List, Optional, Type

from .base import TEXBackendBase, TEXModule


# Global registry
_BACKEND_REGISTRY: Dict[str, Type[TEXBackendBase]] = {}
_BACKEND_INSTANCES: Dict[str, TEXBackendBase] = {}
_CURRENT_BACKEND: Optional[str] = None
_TEX_MODULE: Optional[TEXModule] = None


def register_backend(cls: Type[TEXBackendBase]) -> Type[TEXBackendBase]:
    """
    Decorator to register a backend class.

    Usage:
        @register_backend
        class MyBackend(TEXBackendBase):
            ...
    """
    # Create a temporary instance to get the name
    # We'll use class attributes if available
    if hasattr(cls, 'NAME'):
        name = cls.NAME
    else:
        # Fall back to creating instance (not ideal for heavy backends)
        try:
            instance = cls()
            name = instance.name
        except Exception:
            name = cls.__name__.lower().replace('backend', '')

    if name in _BACKEND_REGISTRY:
        raise ValueError(f"Backend '{name}' is already registered")

    _BACKEND_REGISTRY[name] = cls
    return cls


def _get_backend_instance(name: str) -> TEXBackendBase:
    """Get or create a backend instance."""
    if name not in _BACKEND_INSTANCES:
        if name not in _BACKEND_REGISTRY:
            raise ValueError(
                f"Backend '{name}' not found. "
                f"Available backends: {list(_BACKEND_REGISTRY.keys())}"
            )
        _BACKEND_INSTANCES[name] = _BACKEND_REGISTRY[name]()
    return _BACKEND_INSTANCES[name]


def get_backend(name: Optional[str] = None) -> TEXBackendBase:
    """
    Get a backend by name or auto-select the best available.

    Args:
        name: Backend name. If None, auto-selects based on priority.

    Returns:
        TEXBackendBase instance
    """
    if name is not None:
        return _get_backend_instance(name)

    # Check environment variable
    env_backend = os.environ.get('TE_BACKEND') or os.environ.get('TE_PLUGIN')
    if env_backend:
        return _get_backend_instance(env_backend)

    # Auto-select based on priority and availability
    available_backends = []
    for backend_name, backend_cls in _BACKEND_REGISTRY.items():
        try:
            # Check availability without full instantiation if possible
            if hasattr(backend_cls, 'check_available'):
                if backend_cls.check_available():
                    instance = _get_backend_instance(backend_name)
                    available_backends.append((instance.priority, backend_name, instance))
            else:
                instance = _get_backend_instance(backend_name)
                if instance.is_available():
                    available_backends.append((instance.priority, backend_name, instance))
        except Exception:
            continue

    if not available_backends:
        # Try to get loading errors for better diagnostics
        loading_errors_info = ""
        try:
            backends_module = sys.modules.get("tex_interface.backends")
            if backends_module and hasattr(backends_module, "_loading_errors"):
                errors = backends_module._loading_errors
                if errors:
                    loading_errors_info = "\nBackend loading errors:\n" + "\n".join(
                        f"  - {name}: {err_type}: {msg}" for name, err_type, msg in errors
                    )
        except Exception:
            pass

        raise RuntimeError(
            "No available TEX backends found. "
            f"Registered backends: {list(_BACKEND_REGISTRY.keys())}"
            f"{loading_errors_info}"
        )

    # Sort by priority (highest first) and return the best
    available_backends.sort(key=lambda x: x[0], reverse=True)
    return available_backends[0][2]


def get_current_backend() -> TEXBackendBase:
    """
    Get the currently active backend.

    If no backend is set, auto-selects the best available.
    """
    global _CURRENT_BACKEND

    if _CURRENT_BACKEND is None:
        backend = get_backend()
        _CURRENT_BACKEND = backend.name

    return _get_backend_instance(_CURRENT_BACKEND)


def set_backend(name: str) -> TEXBackendBase:
    """
    Set the active backend by name.

    Args:
        name: Backend name to activate

    Returns:
        The activated backend instance
    """
    global _CURRENT_BACKEND, _TEX_MODULE

    backend = _get_backend_instance(name)
    if not backend.is_available():
        raise RuntimeError(f"Backend '{name}' is not available in current environment")

    _CURRENT_BACKEND = name
    _TEX_MODULE = None  # Reset module so it picks up new backend

    return backend


def list_backends() -> List[Dict]:
    """
    List all registered backends with their status.

    Returns:
        List of dicts with backend info
    """
    result = []
    for name, cls in _BACKEND_REGISTRY.items():
        try:
            instance = _get_backend_instance(name)
            result.append({
                'name': name,
                'vendor': instance.vendor,
                'priority': instance.priority,
                'available': instance.is_available(),
                'current': name == _CURRENT_BACKEND,
            })
        except Exception as e:
            result.append({
                'name': name,
                'vendor': 'Unknown',
                'priority': 0,
                'available': False,
                'current': False,
                'error': str(e),
            })
    return result


def get_tex_module() -> TEXModule:
    """
    Get the TEXModule instance for the current backend.

    This is what gets exported as `transformer_engine_fl_torch`.
    """
    global _TEX_MODULE

    if _TEX_MODULE is None:
        backend = get_current_backend()
        # Pass registry functions so they can be accessed via tex.list_backends() etc.
        registry_funcs = {
            'get_backend': get_backend,
            'set_backend': set_backend,
            'list_backends': list_backends,
        }
        _TEX_MODULE = TEXModule(backend, registry_funcs)

    return _TEX_MODULE


def reset_registry():
    """Reset the registry (mainly for testing)."""
    global _BACKEND_REGISTRY, _BACKEND_INSTANCES, _CURRENT_BACKEND, _TEX_MODULE
    _BACKEND_REGISTRY.clear()
    _BACKEND_INSTANCES.clear()
    _CURRENT_BACKEND = None
    _TEX_MODULE = None
