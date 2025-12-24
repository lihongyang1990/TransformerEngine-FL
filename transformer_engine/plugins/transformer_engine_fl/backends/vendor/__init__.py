# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Vendor-specific backend implementations.

This package contains hardware vendor-specific backend implementations
for TransformerEngine-FL. Each vendor subdirectory should contain its
own backend implementation.
"""

from __future__ import annotations

import os

_DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"


def _log_debug(msg: str) -> None:
    if _DEBUG:
        print(f"[TEFL Vendor] {msg}")


_vendor_loading_errors = []

try:
    from ..._build_config import SKIP_CUDA_BUILD as _SKIP_CUDA_BUILD_CONFIG
    _log_debug(f"Build config loaded: SKIP_CUDA_BUILD={_SKIP_CUDA_BUILD_CONFIG}")
except ImportError:
    _SKIP_CUDA_BUILD_CONFIG = bool(int(os.environ.get("TE_FL_SKIP_CUDA", "0")))
    _log_debug(f"Build config not found, using env var: SKIP_CUDA_BUILD={_SKIP_CUDA_BUILD_CONFIG}")

if os.environ.get("TE_FL_SKIP_CUDA"):
    _SKIP_CUDA_BUILD = bool(int(os.environ.get("TE_FL_SKIP_CUDA", "0")))
    if _DEBUG and _SKIP_CUDA_BUILD != _SKIP_CUDA_BUILD_CONFIG:
        _log_debug(f"Runtime override: SKIP_CUDA_BUILD={_SKIP_CUDA_BUILD} (build-time was {_SKIP_CUDA_BUILD_CONFIG})")
else:
    _SKIP_CUDA_BUILD = _SKIP_CUDA_BUILD_CONFIG

_log_debug("Loading vendor backends...")
_log_debug(f"SKIP_CUDA_BUILD = {_SKIP_CUDA_BUILD}")

if not _SKIP_CUDA_BUILD:
    try:
        from .cuda import CUDABackend
        _log_debug("CUDA vendor backend registered successfully")
    except ImportError as e:
        _vendor_loading_errors.append(("cuda", "ImportError", str(e)))
        _log_debug(f"Failed to import CUDA vendor backend: {e}")
    except Exception as e:
        _vendor_loading_errors.append(("cuda", type(e).__name__, str(e)))
        _log_debug(f"Error loading CUDA vendor backend: {type(e).__name__}: {e}")
        if _DEBUG:
            import traceback
            traceback.print_exc()
else:
    _log_debug("CUDA vendor backend skipped (CUDA build was disabled at build time)")
    _vendor_loading_errors.append(("cuda", "Skipped", "CUDA build was disabled at build time"))


def get_vendor_loading_errors():
    """Get errors that occurred during vendor backend loading."""
    return _vendor_loading_errors.copy()


__all__ = ["get_vendor_loading_errors"]
