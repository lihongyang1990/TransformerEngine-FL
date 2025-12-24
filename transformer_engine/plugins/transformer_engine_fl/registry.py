# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import TEFLBackendBase, TEFLModule

def _get_backend_registry():
    from .backend_registry import BackendRegistry
    return BackendRegistry

def register_backend(cls: Type[TEFLBackendBase]) -> Type[TEFLBackendBase]:
    return _get_backend_registry().get_instance().register_backend(cls)


def unregister_backend(name: str) -> bool:
    return _get_backend_registry().get_instance().unregister_backend(name)


def get_backend(name: Optional[str] = None) -> TEFLBackendBase:
    return _get_backend_registry().get_instance().get_backend(name)


def get_current_backend() -> TEFLBackendBase:
    return _get_backend_registry().get_instance().get_current_backend()


def set_backend(name: str) -> TEFLBackendBase:
    return _get_backend_registry().get_instance().set_backend(name)


def list_backends() -> List[Dict]:
    return _get_backend_registry().get_instance().list_backends()


def get_registered_backend_names() -> List[str]:
    return _get_backend_registry().get_instance().get_registered_backend_names()


def is_backend_registered(name: str) -> bool:
    return _get_backend_registry().get_instance().is_backend_registered(name)


def get_tefl_module() -> TEFLModule:
    return _get_backend_registry().get_instance().get_tefl_module()


def reset_registry() -> None:
    _get_backend_registry().get_instance().reset_registry()


def get_registry_snapshot() -> Dict[str, Type[TEFLBackendBase]]:
    return _get_backend_registry().get_instance().get_registry_snapshot()


def _get_backend_instance(name: str) -> TEFLBackendBase:
    return _get_backend_registry().get_instance()._get_backend_instance(name)


def _export_backend_registry_class():
    from .backend_registry import BackendRegistry
    return BackendRegistry

class _RegistryModule:
    @property
    def BackendRegistry(self):
        return _export_backend_registry_class()

_registry_module = _RegistryModule()
BackendRegistry = _registry_module.BackendRegistry

_BACKEND_REGISTRY: Dict[str, Type[TEFLBackendBase]] = {}
_BACKEND_INSTANCES: Dict[str, TEFLBackendBase] = {}
_CURRENT_BACKEND: Optional[str] = None
_TEFL_MODULE: Optional[TEFLModule] = None
__all__ = [
    "BackendRegistry",
    "register_backend",
    "unregister_backend",
    "get_backend",
    "get_current_backend",
    "set_backend",
    "list_backends",
    "get_registered_backend_names",
    "is_backend_registered",
    "get_registry_snapshot",
    "reset_registry",
    "get_tefl_module",
]
