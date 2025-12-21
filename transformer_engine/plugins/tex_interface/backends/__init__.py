# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
TEX Interface Backends

This package contains backend implementations for different hardware vendors.
Each backend wraps the vendor-specific C++ extensions and provides them
through the unified TEXBackendBase interface.

Available Backends:
- flaggems: FlagGems cross-platform backend (highest priority, supports fallback)
- nvidia: NVIDIA CUDA backend (wraps transformer_engine_torch)
- template: Template for vendors to implement

Usage:
    # Import to register backends
    from transformer_engine.plugins.tex_interface import backends

    # Or import specific backend
    from transformer_engine.plugins.tex_interface.backends.flaggems import FlagGemsBackend
    from transformer_engine.plugins.tex_interface.backends.nvidia import NVIDIABackend
"""

import os
import sys

_DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"

# Load build-time configuration
try:
    from .._build_config import SKIP_CUDA_BUILD as _SKIP_CUDA_BUILD_CONFIG
    if _DEBUG:
        print(f"[TEX] Build config loaded: SKIP_CUDA_BUILD={_SKIP_CUDA_BUILD_CONFIG}")
except ImportError:
    # Fallback to environment variable if config file doesn't exist
    _SKIP_CUDA_BUILD_CONFIG = bool(int(os.environ.get("TE_FL_SKIP_CUDA", "0")))
    if _DEBUG:
        print(f"[TEX] Build config not found, using env var: SKIP_CUDA_BUILD={_SKIP_CUDA_BUILD_CONFIG}")

# Allow runtime override via environment variable (for testing/debugging)
# Build-time config takes precedence, but can be overridden
if os.environ.get("TE_FL_SKIP_CUDA"):
    _SKIP_CUDA_BUILD = bool(int(os.environ.get("TE_FL_SKIP_CUDA", "0")))
    if _DEBUG and _SKIP_CUDA_BUILD != _SKIP_CUDA_BUILD_CONFIG:
        print(f"[TEX] Runtime override: SKIP_CUDA_BUILD={_SKIP_CUDA_BUILD} (build-time was {_SKIP_CUDA_BUILD_CONFIG})")
else:
    _SKIP_CUDA_BUILD = _SKIP_CUDA_BUILD_CONFIG

if _DEBUG:
    print("[TEX] Loading backends...")
    print(f"[TEX] SKIP_CUDA_BUILD = {_SKIP_CUDA_BUILD}")

# Track loading errors for debugging
_loading_errors = []

# Import backends using relative imports (works when imported as transformer_engine.plugins.tex_interface.backends)
# FlagGems first (highest priority, supports fallback to others)
try:
    from .flaggems import FlagGemsBackend  # This triggers @register_backend
    if _DEBUG:
        print("[TEX] FlagGems backend registered successfully")
except ImportError as e:
    _loading_errors.append(("flaggems", "ImportError", str(e)))
    if _DEBUG:
        print(f"[TEX] Failed to import flaggems backend: {e}")
except Exception as e:
    _loading_errors.append(("flaggems", type(e).__name__, str(e)))
    if _DEBUG:
        print(f"[TEX] Error loading flaggems backend: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# PyTorch native backend (pure PyTorch implementation)
try:
    from .torch_backend import torch_backend
    if _DEBUG:
        print("[TEX] Torch backend registered successfully")
except ImportError as e:
    _loading_errors.append(("torch", "ImportError", str(e)))
    if _DEBUG:
        print(f"[TEX] Failed to import torch backend: {e}")
except Exception as e:
    _loading_errors.append(("torch", type(e).__name__, str(e)))
    if _DEBUG:
        print(f"[TEX] Error loading torch backend: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# NVIDIA backend - only load if CUDA was built
if not _SKIP_CUDA_BUILD:
    try:
        from .nvidia import nvidia
        if _DEBUG:
            print("[TEX] NVIDIA backend registered successfully")
    except ImportError as e:
        _loading_errors.append(("nvidia", "ImportError", str(e)))
        if _DEBUG:
            print(f"[TEX] Failed to import nvidia backend: {e}")
    except Exception as e:
        _loading_errors.append(("nvidia", type(e).__name__, str(e)))
        if _DEBUG:
            print(f"[TEX] Error loading nvidia backend: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
else:
    if _DEBUG:
        print("[TEX] NVIDIA backend skipped (CUDA build was disabled at build time)")
    _loading_errors.append(("nvidia", "Skipped", "CUDA build was disabled at build time"))

if _DEBUG:
    # Access registry to show registered backends
    _registry = (
        sys.modules.get("tex_interface.registry") or
        sys.modules.get("transformer_engine.plugins.tex_interface.registry")
    )
    if _registry:
        print(f"[TEX] Registered backends: {list(_registry._BACKEND_REGISTRY.keys())}")

