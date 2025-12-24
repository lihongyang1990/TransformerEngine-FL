# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import os
import sys

_DEBUG = os.environ.get("TE_FL_DEBUG", "0") == "1"


def _log_debug(msg: str) -> None:
    if _DEBUG:
        print(f"[TEFL] {msg}")


_log_debug("Loading backends...")

_loading_errors = []

try:
    from .flagos import FlagOSBackend
    _log_debug("flagos backend registered successfully")
except ImportError as e:
    _loading_errors.append(("flagos", "ImportError", str(e)))
    _log_debug(f"Failed to import flagos backend: {e}")
except Exception as e:
    _loading_errors.append(("flagos", type(e).__name__, str(e)))
    _log_debug(f"Error loading flagos backend: {type(e).__name__}: {e}")
    if _DEBUG:
        import traceback
        traceback.print_exc()

try:
    from .reference import reference
    _log_debug("reference backend registered successfully")
except ImportError as e:
    _loading_errors.append(("reference", "ImportError", str(e)))
    _log_debug(f"Failed to import reference backend: {e}")
except Exception as e:
    _loading_errors.append(("reference", type(e).__name__, str(e)))
    _log_debug(f"Error loading reference backend: {type(e).__name__}: {e}")
    if _DEBUG:
        import traceback
        traceback.print_exc()

try:
    from . import vendor
    _log_debug("Vendor backends module loaded")
    vendor_errors = vendor.get_vendor_loading_errors()
    if vendor_errors:
        _loading_errors.extend(vendor_errors)
except ImportError as e:
    _loading_errors.append(("vendor", "ImportError", str(e)))
    _log_debug(f"Failed to import vendor backends: {e}")
except Exception as e:
    _loading_errors.append(("vendor", type(e).__name__, str(e)))
    _log_debug(f"Error loading vendor backends: {type(e).__name__}: {e}")
    if _DEBUG:
        import traceback
        traceback.print_exc()

_log_debug("Starting plugin discovery...")

try:
    from ..discovery import discover_plugins

    _registry = (
        sys.modules.get("transformer_engine.plugins.transformer_engine_fl.registry") or
        sys.modules.get("transformer_engine_fl.registry")
    )
    if _registry is None:
        try:
            from .. import registry as _registry
        except ImportError:
            _registry = None

    if _registry is not None:
        plugins_loaded = discover_plugins(_registry)
        _log_debug(f"Plugin discovery complete: {plugins_loaded} plugins loaded")
    else:
        _log_debug("Registry module not available, skipping plugin discovery")

except ImportError as e:
    _log_debug(f"Discovery module not available: {e}")
except Exception as e:
    _log_debug(f"Error during plugin discovery: {type(e).__name__}: {e}")
    if _DEBUG:
        import traceback
        traceback.print_exc()

if _DEBUG:
    try:
        from .. import registry as _registry
        backend_names = _registry.get_registered_backend_names()
        _log_debug(f"Registered backends: {backend_names}")
    except Exception as e:
        _log_debug(f"Could not access registry: {e}")

    if _loading_errors:
        _log_debug("Backend loading errors:")
        for name, err_type, msg in _loading_errors:
            _log_debug(f"  - {name}: {err_type}: {msg}")


def get_loading_errors():
    return _loading_errors.copy()


def reload_backends():
    global _loading_errors
    _loading_errors.clear()

    try:
        from ..discovery import discover_plugins, clear_discovered_plugins

        clear_discovered_plugins()

        _registry = (
            sys.modules.get("transformer_engine.plugins.transformer_engine_fl.registry") or
            sys.modules.get("transformer_engine_fl.registry")
        )
        if _registry is not None:
            discover_plugins(_registry)
            _log_debug("Backends reloaded")

    except Exception as e:
        _log_debug(f"Error reloading backends: {e}")
