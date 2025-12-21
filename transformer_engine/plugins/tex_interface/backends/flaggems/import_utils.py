# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Import utilities for FlagGems backend.
"""


def have_flag_gems() -> bool:
    """
    Check if flag_gems library is available.

    Returns:
        bool: True if flag_gems can be imported, False otherwise
    """
    try:
        import flag_gems  # noqa: F401
        return True
    except ImportError:
        return False
