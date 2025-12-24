from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .base import TEFLBackendBase, TEFLModule

logger = logging.getLogger("transformer_engine_fl.backend_registry")

_DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"


def _log_debug(msg: str) -> None:
    if _DEBUG:
        print(f"[TEFL Registry] {msg}")
    logger.debug(msg)


@dataclass
class _RegistryState:
    registry: Dict[str, Type[TEFLBackendBase]]
    instances: Dict[str, TEFLBackendBase]
    current_backend: Optional[str]
    tefl_module: Optional[TEFLModule]
    init_pid: int
    initialized: bool

    def __init__(self):
        self.registry = {}
        self.instances = {}
        self.current_backend = None
        self.tefl_module = None
        self.init_pid = -1
        self.initialized = False


class BackendRegistry:

    _instance = None
    _lock = threading.RLock()

    def __init__(self):
        if hasattr(self, '_state'):
            return

        self._state = _RegistryState()
        self._state.init_pid = os.getpid()

        try:
            os.register_at_fork(after_in_child=self._reset_after_fork)
            _log_debug("Registered at_fork handler")
        except AttributeError:
            _log_debug("os.register_at_fork not available, using PID check")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__()
        return cls._instance

    def _reset_after_fork(self) -> None:
        with self._lock:
            _log_debug("Resetting registry state after fork")
            self._state.instances.clear()
            self._state.current_backend = None
            self._state.tefl_module = None
            self._state.init_pid = os.getpid()
            self._state.initialized = False

            try:
                from .policy import bump_policy_epoch
                bump_policy_epoch()
            except ImportError:
                pass

    def _check_fork(self) -> None:
        pid = os.getpid()
        if self._state.init_pid != -1 and self._state.init_pid != pid:
            self._reset_after_fork()

    def register_backend(self, cls: Type[TEFLBackendBase]) -> Type[TEFLBackendBase]:
        self._check_fork()

        if hasattr(cls, 'NAME'):
            name = cls.NAME
        else:
            try:
                instance = cls()
                name = instance.name
            except Exception:
                name = cls.__name__.lower().replace('backend', '')

        with self._lock:
            if name in self._state.registry:
                raise ValueError(f"Backend '{name}' is already registered")

            self._state.registry[name] = cls
            _log_debug(f"Registered backend: {name}")

        return cls

    def unregister_backend(self, name: str) -> bool:
        self._check_fork()

        with self._lock:
            if name not in self._state.registry:
                return False

            del self._state.registry[name]
            self._state.instances.pop(name, None)

            if self._state.current_backend == name:
                self._state.current_backend = None
                self._state.tefl_module = None

            _log_debug(f"Unregistered backend: {name}")

        return True

    def _get_backend_instance(self, name: str) -> TEFLBackendBase:
        self._check_fork()

        with self._lock:
            if name not in self._state.instances:
                if name not in self._state.registry:
                    raise ValueError(
                        f"Backend '{name}' not found. "
                        f"Available backends: {list(self._state.registry.keys())}"
                    )
                self._state.instances[name] = self._state.registry[name]()
                _log_debug(f"Created instance for backend: {name}")

        return self._state.instances[name]

    def get_backend(self, name: Optional[str] = None) -> TEFLBackendBase:
        self._check_fork()

        if name is not None:
            return self._get_backend_instance(name)

        env_backend = os.environ.get('TE_FL_BACKEND') or os.environ.get('TE_PLUGIN')
        if env_backend:
            return self._get_backend_instance(env_backend)

        try:
            from .policy import get_policy
            policy = get_policy()
        except ImportError:
            policy = None

        available_backends = []

        with self._lock:
            registry_copy = dict(self._state.registry)

        for backend_name, backend_cls in registry_copy.items():
            try:
                if hasattr(backend_cls, 'check_available'):
                    if not backend_cls.check_available():
                        continue

                instance = self._get_backend_instance(backend_name)

                if not instance.is_available():
                    continue

                if policy is not None:
                    vendor = getattr(instance, 'vendor', None)
                    if vendor and not policy.is_vendor_allowed(vendor):
                        _log_debug(f"Backend {backend_name} denied by policy")
                        continue

                available_backends.append((instance.priority, backend_name, instance))

            except Exception as e:
                _log_debug(f"Error checking backend {backend_name}: {e}")
                continue

        if not available_backends:
            loading_errors_info = ""
            try:
                backends_module = (
                    sys.modules.get("transformer_engine.plugins.transformer_engine_fl.backends") or
                    sys.modules.get("transformer_engine_fl.backends")
                )
                if backends_module and hasattr(backends_module, "_loading_errors"):
                    errors = backends_module._loading_errors
                    if errors:
                        loading_errors_info = "\nBackend loading errors:\n" + "\n".join(
                            f" - {name}: {err_type}: {msg}" for name, err_type, msg in errors
                        )
            except Exception:
                pass

            raise RuntimeError(
                "No available TEFL backends found. "
                f"Registered backends: {list(self._state.registry.keys())}"
                f"{loading_errors_info}"
            )

        available_backends.sort(key=lambda x: x[0], reverse=True)
        return available_backends[0][2]

    def get_current_backend(self) -> TEFLBackendBase:
        self._check_fork()

        with self._lock:
            if self._state.current_backend is None:
                backend = self.get_backend()
                self._state.current_backend = backend.name

        return self._get_backend_instance(self._state.current_backend)

    def set_backend(self, name: str) -> TEFLBackendBase:
        self._check_fork()

        with self._lock:
            backend = self._get_backend_instance(name)
            if not backend.is_available():
                raise RuntimeError(f"Backend '{name}' is not available in current environment")

            self._state.current_backend = name
            self._state.tefl_module = None

            try:
                from .policy import bump_policy_epoch
                bump_policy_epoch()
            except ImportError:
                pass

            _log_debug(f"Set current backend to: {name}")

        return backend

    def list_backends(self) -> List[Dict]:
        self._check_fork()

        result = []

        with self._lock:
            registry_copy = dict(self._state.registry)
            current = self._state.current_backend

        for name, cls in registry_copy.items():
            try:
                instance = self._get_backend_instance(name)
                info = {
                    'name': name,
                    'vendor': instance.vendor,
                    'priority': instance.priority,
                    'available': instance.is_available(),
                    'current': name == current,
                }
                if hasattr(instance, 'impl_kind'):
                    info['impl_kind'] = str(instance.impl_kind.value)
                result.append(info)
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

    def get_registered_backend_names(self) -> List[str]:
        self._check_fork()

        with self._lock:
            return list(self._state.registry.keys())

    def is_backend_registered(self, name: str) -> bool:
        self._check_fork()

        with self._lock:
            return name in self._state.registry

    def get_tefl_module(self) -> TEFLModule:
        self._check_fork()

        with self._lock:
            if self._state.tefl_module is None:
                backend = self.get_current_backend()
                registry_funcs = {
                    'get_backend': self.get_backend,
                    'set_backend': self.set_backend,
                    'list_backends': self.list_backends,
                }
                self._state.tefl_module = TEFLModule(backend, registry_funcs)

        return self._state.tefl_module

    def reset_registry(self) -> None:
        with self._lock:
            self._state.registry.clear()
            self._state.instances.clear()
            self._state.current_backend = None
            self._state.tefl_module = None
            self._state.init_pid = os.getpid()
            self._state.initialized = False

            try:
                from .policy import bump_policy_epoch
                bump_policy_epoch()
            except ImportError:
                pass

            _log_debug("Registry reset")

    def get_registry_snapshot(self) -> Dict[str, Type[TEFLBackendBase]]:
        self._check_fork()

        with self._lock:
            return dict(self._state.registry)
