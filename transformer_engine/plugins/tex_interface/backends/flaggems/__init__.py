# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""FlagGems Backend Package."""

# Import to trigger @register_backend decorator
from .flaggems import FlagGemsBackend

# Export for convenience
__all__ = ["FlagGemsBackend"]
