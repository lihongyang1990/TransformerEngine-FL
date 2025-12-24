# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
CUDA vendor backend operator registrations.

This module registers all VENDOR (CUDA) implementations from transformer_engine_torch.
"""

from __future__ import annotations

from ....types import OpImpl, BackendImplKind


def register_builtins(registry) -> None:
    """
    Register all CUDA (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    # Import CUDA backend to get all the wrapped tex functions
    from .cuda import CUDABackend
    
    # Create a backend instance to access the methods
    backend = CUDABackend()
    
    # Check if CUDA is available before registering
    if not backend.is_available():
        return
    
    impls = [
        # Normalization
        OpImpl(op_name="rmsnorm_fwd", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.rmsnorm_fwd, vendor="CUDA", priority=100),
        OpImpl(op_name="rmsnorm_bwd", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.rmsnorm_bwd, vendor="CUDA", priority=100),
        OpImpl(op_name="layernorm_fwd", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.layernorm_fwd, vendor="CUDA", priority=100),
        OpImpl(op_name="layernorm_bwd", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.layernorm_bwd, vendor="CUDA", priority=100),
        
        # GEMM
        OpImpl(op_name="generic_gemm", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.generic_gemm, vendor="CUDA", priority=100),
        
        # Quantization
        OpImpl(op_name="quantize", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.quantize, vendor="CUDA", priority=100),
        OpImpl(op_name="dequantize", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dequantize, vendor="CUDA", priority=100),
        OpImpl(op_name="bgrad_quantize", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.bgrad_quantize, vendor="CUDA", priority=100),
        
        # Activations - Forward
        OpImpl(op_name="gelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.gelu, vendor="CUDA", priority=100),
        OpImpl(op_name="geglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.geglu, vendor="CUDA", priority=100),
        OpImpl(op_name="qgelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.qgelu, vendor="CUDA", priority=100),
        OpImpl(op_name="qgeglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.qgeglu, vendor="CUDA", priority=100),
        OpImpl(op_name="relu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.relu, vendor="CUDA", priority=100),
        OpImpl(op_name="reglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.reglu, vendor="CUDA", priority=100),
        OpImpl(op_name="srelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.srelu, vendor="CUDA", priority=100),
        OpImpl(op_name="sreglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.sreglu, vendor="CUDA", priority=100),
        OpImpl(op_name="silu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.silu, vendor="CUDA", priority=100),
        OpImpl(op_name="swiglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.swiglu, vendor="CUDA", priority=100),
        OpImpl(op_name="clamped_swiglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.clamped_swiglu, vendor="CUDA", priority=100),
        
        # Activations - Backward
        OpImpl(op_name="dgelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dgelu, vendor="CUDA", priority=100),
        OpImpl(op_name="dgeglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dgeglu, vendor="CUDA", priority=100),
        OpImpl(op_name="dqgelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dqgelu, vendor="CUDA", priority=100),
        OpImpl(op_name="dqgeglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dqgeglu, vendor="CUDA", priority=100),
        OpImpl(op_name="drelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.drelu, vendor="CUDA", priority=100),
        OpImpl(op_name="dreglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dreglu, vendor="CUDA", priority=100),
        OpImpl(op_name="dsrelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dsrelu, vendor="CUDA", priority=100),
        OpImpl(op_name="dsreglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dsreglu, vendor="CUDA", priority=100),
        OpImpl(op_name="dsilu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dsilu, vendor="CUDA", priority=100),
        OpImpl(op_name="dswiglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dswiglu, vendor="CUDA", priority=100),
        OpImpl(op_name="clamped_dswiglu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.clamped_dswiglu, vendor="CUDA", priority=100),
        
        # Activations - Bias + Backward
        OpImpl(op_name="dbias_dgelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dbias_dgelu, vendor="CUDA", priority=100),
        OpImpl(op_name="dbias_dsilu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dbias_dsilu, vendor="CUDA", priority=100),
        OpImpl(op_name="dbias_drelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dbias_drelu, vendor="CUDA", priority=100),
        OpImpl(op_name="dbias_dqgelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dbias_dqgelu, vendor="CUDA", priority=100),
        OpImpl(op_name="dbias_dsrelu", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dbias_dsrelu, vendor="CUDA", priority=100),
        
        # Softmax
        OpImpl(op_name="scaled_softmax_forward", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.scaled_softmax_forward, vendor="CUDA", priority=100),
        OpImpl(op_name="scaled_softmax_backward", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.scaled_softmax_backward, vendor="CUDA", priority=100),
        OpImpl(op_name="scaled_masked_softmax_forward", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.scaled_masked_softmax_forward, vendor="CUDA", priority=100),
        OpImpl(op_name="scaled_masked_softmax_backward", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.scaled_masked_softmax_backward, vendor="CUDA", priority=100),
        OpImpl(op_name="scaled_upper_triang_masked_softmax_forward", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.scaled_upper_triang_masked_softmax_forward, vendor="CUDA", priority=100),
        OpImpl(op_name="scaled_upper_triang_masked_softmax_backward", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.scaled_upper_triang_masked_softmax_backward, vendor="CUDA", priority=100),
        OpImpl(op_name="scaled_aligned_causal_masked_softmax_forward", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.scaled_aligned_causal_masked_softmax_forward, vendor="CUDA", priority=100),
        OpImpl(op_name="scaled_aligned_causal_masked_softmax_backward", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.scaled_aligned_causal_masked_softmax_backward, vendor="CUDA", priority=100),
        
        # Dropout
        OpImpl(op_name="dropout_fwd", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dropout_fwd, vendor="CUDA", priority=100),
        OpImpl(op_name="dropout_bwd", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.dropout_bwd, vendor="CUDA", priority=100),

        # Multi-tensor operations
        OpImpl(op_name="multi_tensor_quantize", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_quantize, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_scale", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_scale, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_l2norm", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_l2norm, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_unscale_l2norm", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_unscale_l2norm, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_adam", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_adam, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_adam_param_remainder", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_adam_param_remainder, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_adam_fp8", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_adam_fp8, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_adam_capturable", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_adam_capturable, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_adam_capturable_master", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_adam_capturable_master, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_sgd", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_sgd, vendor="CUDA", priority=100),
        OpImpl(op_name="multi_tensor_compute_scale_and_scale_inv", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.multi_tensor_compute_scale_and_scale_inv, vendor="CUDA", priority=100),

        # FlashAttention class getter
        OpImpl(op_name="get_flash_attention_class", impl_id="vendor.cuda", kind=BackendImplKind.VENDOR, fn=backend.get_flash_attention_class, vendor="CUDA", priority=100),
    ]

    registry.register_many(impls)
