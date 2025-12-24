from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import TEFLBackendBase


_DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"


def _log_debug(msg: str) -> None:
    if _DEBUG:
        print(f"[OpManager] {msg}")


class OperatorManager:

    _instance = None
    _lock = threading.RLock()

    def __init__(self):
        if hasattr(self, '_operator_backend_map'):
            return

        self._operator_backend_map: Dict[str, str] = {}

        self._operator_priority_map: Dict[str, Dict[str, int]] = {}

        self._operator_impl_cache: Dict[Tuple[str, str, int], Tuple[str, Callable]] = {}

        self._local_epoch: int = 0

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__()
                    cls._instance.configure_from_env()
        return cls._instance

    def _bump_local_epoch(self) -> None:
        self._local_epoch += 1

    def _get_cache_key(self, op_name: str) -> Tuple[str, str, int]:
        try:
            from .policy import get_policy, get_policy_epoch
            policy = get_policy()
            policy_epoch = get_policy_epoch()
            fingerprint = policy.fingerprint()
        except ImportError:
            fingerprint = "no_policy"
            policy_epoch = 0

        return (op_name, fingerprint, policy_epoch + self._local_epoch)

    def set_operator_backend(self, op_name: str, backend_name: str) -> None:
        with self._lock:
            self._operator_backend_map[op_name] = backend_name
            self._bump_local_epoch()
            _log_debug(f"Set {op_name} -> {backend_name}")

    def set_operator_priority(self, op_name: str, priorities: Dict[str, int]) -> None:
        with self._lock:
            self._operator_priority_map[op_name] = priorities.copy()
            self._bump_local_epoch()
            _log_debug(f"Set priorities for {op_name}: {priorities}")

    def get_operator_backend(self, op_name: str) -> Optional[str]:
        env_key = f"TE_OP_BACKEND_{op_name}"
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val

        with self._lock:
            return self._operator_backend_map.get(op_name)

    def get_operator_priority(self, op_name: str) -> Optional[Dict[str, int]]:
        env_key = f"TE_OP_PRIORITY_{op_name}"
        env_val = os.environ.get(env_key)
        if env_val:
            priorities = {}
            for item in env_val.split(","):
                parts = item.strip().split(":")
                if len(parts) == 2:
                    backend, priority = parts
                    try:
                        priorities[backend.strip()] = int(priority.strip())
                    except ValueError:
                        pass
            if priorities:
                return priorities

        with self._lock:
            prio = self._operator_priority_map.get(op_name)
            return prio.copy() if prio else None

    def clear_operator_config(self, op_name: Optional[str] = None) -> None:
        with self._lock:
            if op_name:
                self._operator_backend_map.pop(op_name, None)
                self._operator_priority_map.pop(op_name, None)
            else:
                self._operator_backend_map.clear()
                self._operator_priority_map.clear()

            self._operator_impl_cache.clear()
            self._bump_local_epoch()

    def list_operator_config(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "explicit_backends": self._operator_backend_map.copy(),
                "priority_configs": {k: v.copy() for k, v in self._operator_priority_map.items()},
            }

    def configure_operators(self, config: Dict[str, Any]) -> None:
        with self._lock:
            for op_name, value in config.items():
                if isinstance(value, str):
                    self._operator_backend_map[op_name] = value
                elif isinstance(value, dict):
                    self._operator_priority_map[op_name] = value.copy()
                else:
                    raise ValueError(f"Invalid config for {op_name}: {value}")

            self._operator_impl_cache.clear()
            self._bump_local_epoch()

    def configure_from_env(self) -> None:
        with self._lock:
            for key, value in os.environ.items():
                if key.startswith("TE_OP_BACKEND_"):
                    op_name = key[len("TE_OP_BACKEND_"):]
                    self._operator_backend_map[op_name] = value
                elif key.startswith("TE_OP_PRIORITY_"):
                    op_name = key[len("TE_OP_PRIORITY_"):]
                    priorities = {}
                    for item in value.split(","):
                        parts = item.strip().split(":")
                        if len(parts) == 2:
                            backend, priority = parts
                            try:
                                priorities[backend.strip()] = int(priority.strip())
                            except ValueError:
                                pass
                    if priorities:
                        self._operator_priority_map[op_name] = priorities

            self._bump_local_epoch()

    def get_available_backends(self) -> Dict[str, TEFLBackendBase]:
        _reg = (
            sys.modules.get("transformer_engine.plugins.transformer_engine_fl.registry") or
            sys.modules.get("transformer_engine_fl.registry")
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
        self,
        op_name: str,
        available_backends: Optional[Dict[str, TEFLBackendBase]] = None,
    ) -> Tuple[Optional[str], Optional[Callable]]:
        cache_key = self._get_cache_key(op_name)

        with self._lock:
            cached = self._operator_impl_cache.get(cache_key)
            if cached is not None:
                return cached

        if available_backends is None:
            available_backends = self.get_available_backends()

        if not available_backends:
            return None, None

        try:
            from .policy import get_policy
            policy = get_policy()
            strict = policy.strict
        except ImportError:
            policy = None
            strict = False

        explicit_backend = self.get_operator_backend(op_name)
        if explicit_backend:
            if explicit_backend in available_backends:
                backend = available_backends[explicit_backend]
                method = getattr(backend, op_name, None)
                if method is not None:
                    result = (explicit_backend, method)
                    with self._lock:
                        self._operator_impl_cache[cache_key] = result
                    _log_debug(f"{op_name} -> {explicit_backend} (explicit)")
                    return result
            _log_debug(f"Warning: explicit backend {explicit_backend} not available for {op_name}")

        if policy is not None:
            filtered_backends = {}
            for name, backend in available_backends.items():
                vendor = getattr(backend, 'vendor', None)
                if vendor and not policy.is_vendor_allowed(vendor):
                    _log_debug(f"Backend {name} denied by policy for {op_name}")
                    continue
                filtered_backends[name] = backend
            available_backends = filtered_backends

        if not available_backends:
            if strict:
                raise RuntimeError(f"No implementation available for op={op_name} under strict policy (all vendors denied)")
            return None, None

        priorities = self.get_operator_priority(op_name)
        if priorities:
            sorted_backends = sorted(
                [(name, prio) for name, prio in priorities.items() if name in available_backends],
                key=lambda x: x[1],
                reverse=True
            )
            for backend_name, priority in sorted_backends:
                backend = available_backends[backend_name]
                method = getattr(backend, op_name, None)
                if method is not None:
                    result = (backend_name, method)
                    with self._lock:
                        self._operator_impl_cache[cache_key] = result
                    _log_debug(f"{op_name} -> {backend_name} (priority={priority})")
                    return result

        if policy is not None:
            per_op_order = policy.get_per_op_order(op_name)
            if per_op_order:
                for token in per_op_order:
                    matching_backends = []
                    for backend_name, backend in available_backends.items():
                        if self._match_token(backend, token):
                            method = getattr(backend, op_name, None)
                            if method is not None:
                                matching_backends.append((backend.priority, backend_name, method))

                    if matching_backends:
                        matching_backends.sort(key=lambda x: x[0], reverse=True)
                        _, backend_name, method = matching_backends[0]
                        result = (backend_name, method)
                        with self._lock:
                            self._operator_impl_cache[cache_key] = result
                        _log_debug(f"{op_name} -> {backend_name} (policy per-op token={token})")
                        return result

        sorted_backends = sorted(
            available_backends.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        for backend_name, backend in sorted_backends:
            method = getattr(backend, op_name, None)
            if method is not None:
                result = (backend_name, method)
                with self._lock:
                    self._operator_impl_cache[cache_key] = result
                _log_debug(f"{op_name} -> {backend_name} (default priority={backend.priority})")
                return result

        if strict:
            raise RuntimeError(f"No implementation available for op={op_name} under strict policy")

        return None, None

    def _match_token(self, backend: TEFLBackendBase, token: str) -> bool:
        try:
            from .types import BackendImplKind

            impl_kind = getattr(backend, 'impl_kind', None)

            if token == "default":
                return impl_kind == BackendImplKind.DEFAULT
            elif token == "reference":
                return impl_kind == BackendImplKind.REFERENCE
            elif token == "vendor":
                return impl_kind == BackendImplKind.VENDOR
            elif token.startswith("vendor:"):
                vendor_name = token.split(":", 1)[1]
                return getattr(backend, 'vendor', None) == vendor_name

        except ImportError:
            pass

        return getattr(backend, 'name', None) == token

    def call_operator(
        self,
        op_name: str,
        *args,
        available_backends: Optional[Dict[str, TEFLBackendBase]] = None,
        **kwargs
    ) -> Any:
        backend_name, method = self.resolve_operator_impl(op_name, available_backends)

        if method is None:
            raise NotImplementedError(
                f"No implementation found for operator '{op_name}'. "
                f"Available backends: {list(self.get_available_backends().keys())}"
            )

        return method(*args, **kwargs)

    def clear_cache(self) -> None:
        with self._lock:
            self._operator_impl_cache.clear()
            self._bump_local_epoch()

    def get_cache_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._operator_impl_cache),
                "local_epoch": self._local_epoch,
            }
