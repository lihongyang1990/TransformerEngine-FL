# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from .transformer_engine_fl import (
    TEFLBackendBase,
    TEFLModule,
    register_backend,
    get_backend,
    get_current_backend,
    set_backend,
    list_backends,
)

from .transformer_engine_fl.registry import get_tefl_module as _get_tefl_module

def __getattr__(name):
    if name == "tefl":
        return _get_tefl_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TEFLBackendBase",
    "TEFLModule",
    "register_backend",
    "get_backend",
    "get_current_backend",
    "set_backend",
    "list_backends",
    "tefl",
]
