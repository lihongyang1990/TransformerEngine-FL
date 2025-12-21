# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""NVIDIA Backend Package."""

# Import to trigger @register_backend decorator
from .nvidia import NVIDIABackend

# Export for convenience
__all__ = ["NVIDIABackend"]
