# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
FlagOS backend operator registrations.

This module registers all DEFAULT (FlagOS) implementations.
"""

from __future__ import annotations

from ...types import OpImpl, BackendImplKind


def register_builtins(registry) -> None:
    """
    Register all FlagOS (DEFAULT) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .flagos import FlagOSBackend

    # Create a backend instance to access the methods
    backend = FlagOSBackend()

    impls = [
        OpImpl(op_name="rmsnorm_fwd", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=backend.rmsnorm_fwd, vendor=None, priority=150),
        OpImpl(op_name="rmsnorm_bwd", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=backend.rmsnorm_bwd, vendor=None, priority=150),
        OpImpl(op_name="generic_gemm", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=backend.generic_gemm, vendor=None, priority=150),
        OpImpl(op_name="multi_tensor_scale", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=backend.multi_tensor_scale, vendor=None, priority=150),
        OpImpl(op_name="multi_tensor_adam", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=backend.multi_tensor_adam, vendor=None, priority=150),
        OpImpl(op_name="multi_tensor_l2norm", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=backend.multi_tensor_l2norm, vendor=None, priority=150),

        # FlashAttention class getter
        OpImpl(op_name="get_flash_attention_class", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=backend.get_flash_attention_class, vendor=None, priority=150),
    ]
    
    registry.register_many(impls)
