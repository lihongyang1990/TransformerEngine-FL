# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch Native Backend Package."""

# Import to trigger @register_backend decorator
from .torch_backend import TorchBackend

# Export for convenience
__all__ = ["TorchBackend"]
