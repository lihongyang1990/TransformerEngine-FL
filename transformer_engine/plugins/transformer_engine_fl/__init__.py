# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from .types import BackendImplKind, OpImpl, match_token

from .base import TEFLBackendBase, TEFLModule

from .logger_manager import Logger, LoggerManager
from .policy_manager import SelectionPolicy, PolicyManager
from .backend_registry import BackendRegistry
from .operator_manager import OperatorManager

from .registry import (
    register_backend,
    unregister_backend,
    get_backend,
    get_current_backend,
    set_backend,
    list_backends,
    get_registered_backend_names,
    is_backend_registered,
    get_registry_snapshot,
    reset_registry,
)

from .policy import (
    SelectionPolicy,
    get_policy,
    set_global_policy,
    reset_global_policy,
    policy_context,
    policy_from_env,
    get_policy_epoch,
    bump_policy_epoch,
    with_strict_mode,
    with_vendor_preference,
    with_allowed_vendors,
    with_denied_vendors,
)

from .decorators import (
    with_fallback,
    with_debug,
    with_operator_priority,
    with_strict_check,
    with_policy_check,
    with_full_dispatch,
    create_dispatch_decorator,
)

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
    get_available_backends,
    clear_cache,
    get_cache_stats,
)

from .discovery import (
    discover_plugins,
    discover_from_entry_points,
    discover_from_env_modules,
    get_discovered_plugins,
    clear_discovered_plugins,
    PLUGIN_GROUP,
    PLUGIN_MODULES_ENV,
)

# Setup module aliases BEFORE importing backends to support relative imports
from ._module_setup import setup_module_aliases, register_as_transformer_engine_torch
setup_module_aliases()

# Import backends - this loads all available backends (flagos, reference, vendor/cuda, etc.)
from . import backends

# Register transformer_engine_torch AFTER backends are loaded
# so that get_tefl_module() can find a registered backend
register_as_transformer_engine_torch()

__all__ = [
    "BackendImplKind",
    "OpImpl",
    "match_token",
    "TEFLBackendBase",
    "TEFLModule",
    "Logger",
    "LoggerManager",
    "SelectionPolicy",
    "PolicyManager",
    "BackendRegistry",
    "OperatorManager",
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
    "get_policy",
    "set_global_policy",
    "reset_global_policy",
    "policy_context",
    "policy_from_env",
    "get_policy_epoch",
    "bump_policy_epoch",
    "with_strict_mode",
    "with_vendor_preference",
    "with_allowed_vendors",
    "with_denied_vendors",
    "with_fallback",
    "with_debug",
    "with_operator_priority",
    "with_strict_check",
    "with_policy_check",
    "with_full_dispatch",
    "create_dispatch_decorator",
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
    "discover_plugins",
    "discover_from_entry_points",
    "discover_from_env_modules",
    "get_discovered_plugins",
    "clear_discovered_plugins",
    "PLUGIN_GROUP",
    "PLUGIN_MODULES_ENV",
]
