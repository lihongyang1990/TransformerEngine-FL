# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Reference backend operator registrations.

This module registers all REFERENCE (PyTorch) implementations.
"""

from __future__ import annotations

from ...types import OpImpl, BackendImplKind


def register_builtins(registry) -> None:
    """
    Register all PyTorch (REFERENCE) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .reference import ReferenceBackend

    # Create a backend instance to access the methods
    backend = ReferenceBackend()

    impls = [
        # Normalization
        OpImpl(op_name="rmsnorm_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.rmsnorm_fwd, vendor=None, priority=50),
        OpImpl(op_name="rmsnorm_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.rmsnorm_bwd, vendor=None, priority=50),
        OpImpl(op_name="rmsnorm_bwd_add", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.rmsnorm_bwd_add, vendor=None, priority=50),
        OpImpl(op_name="layernorm_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.layernorm_fwd, vendor=None, priority=50),
        OpImpl(op_name="layernorm_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.layernorm_bwd, vendor=None, priority=50),

        # GEMM
        OpImpl(op_name="generic_gemm", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.generic_gemm, vendor=None, priority=50),
        OpImpl(op_name="te_general_grouped_gemm", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.te_general_grouped_gemm, vendor=None, priority=50),

        # Quantization
        OpImpl(op_name="quantize", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.quantize, vendor=None, priority=50),
        OpImpl(op_name="dequantize", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dequantize, vendor=None, priority=50),
        OpImpl(op_name="bgrad_quantize", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.bgrad_quantize, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_quantize", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_quantize, vendor=None, priority=50),
        OpImpl(op_name="split_quantize", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.split_quantize, vendor=None, priority=50),

        # Activations - Forward
        OpImpl(op_name="gelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.gelu, vendor=None, priority=50),
        OpImpl(op_name="geglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.geglu, vendor=None, priority=50),
        OpImpl(op_name="qgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.qgelu, vendor=None, priority=50),
        OpImpl(op_name="qgeglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.qgeglu, vendor=None, priority=50),
        OpImpl(op_name="relu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.relu, vendor=None, priority=50),
        OpImpl(op_name="reglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.reglu, vendor=None, priority=50),
        OpImpl(op_name="srelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.srelu, vendor=None, priority=50),
        OpImpl(op_name="sreglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.sreglu, vendor=None, priority=50),
        OpImpl(op_name="silu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.silu, vendor=None, priority=50),
        OpImpl(op_name="swiglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.swiglu, vendor=None, priority=50),
        OpImpl(op_name="clamped_swiglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.clamped_swiglu, vendor=None, priority=50),

        # Activations - Backward
        OpImpl(op_name="dgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dgelu, vendor=None, priority=50),
        OpImpl(op_name="dgeglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dgeglu, vendor=None, priority=50),
        OpImpl(op_name="dqgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dqgelu, vendor=None, priority=50),
        OpImpl(op_name="dqgeglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dqgeglu, vendor=None, priority=50),
        OpImpl(op_name="drelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.drelu, vendor=None, priority=50),
        OpImpl(op_name="dreglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dreglu, vendor=None, priority=50),
        OpImpl(op_name="dsrelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dsrelu, vendor=None, priority=50),
        OpImpl(op_name="dsreglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dsreglu, vendor=None, priority=50),
        OpImpl(op_name="dsilu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dsilu, vendor=None, priority=50),
        OpImpl(op_name="dswiglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dswiglu, vendor=None, priority=50),
        OpImpl(op_name="clamped_dswiglu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.clamped_dswiglu, vendor=None, priority=50),

        # Activations - Bias + Backward
        OpImpl(op_name="dbias_dgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dbias_dgelu, vendor=None, priority=50),
        OpImpl(op_name="dbias_dsilu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dbias_dsilu, vendor=None, priority=50),
        OpImpl(op_name="dbias_drelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dbias_drelu, vendor=None, priority=50),
        OpImpl(op_name="dbias_dqgelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dbias_dqgelu, vendor=None, priority=50),
        OpImpl(op_name="dbias_dsrelu", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dbias_dsrelu, vendor=None, priority=50),

        # Softmax
        OpImpl(op_name="scaled_softmax_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.scaled_softmax_forward, vendor=None, priority=50),
        OpImpl(op_name="scaled_softmax_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.scaled_softmax_backward, vendor=None, priority=50),
        OpImpl(op_name="scaled_masked_softmax_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.scaled_masked_softmax_forward, vendor=None, priority=50),
        OpImpl(op_name="scaled_masked_softmax_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.scaled_masked_softmax_backward, vendor=None, priority=50),
        OpImpl(op_name="scaled_upper_triang_masked_softmax_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.scaled_upper_triang_masked_softmax_forward, vendor=None, priority=50),
        OpImpl(op_name="scaled_upper_triang_masked_softmax_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.scaled_upper_triang_masked_softmax_backward, vendor=None, priority=50),
        OpImpl(op_name="scaled_aligned_causal_masked_softmax_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.scaled_aligned_causal_masked_softmax_forward, vendor=None, priority=50),
        OpImpl(op_name="scaled_aligned_causal_masked_softmax_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.scaled_aligned_causal_masked_softmax_backward, vendor=None, priority=50),

        # MOE operations
        OpImpl(op_name="moe_permute_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.moe_permute_fwd, vendor=None, priority=50),
        OpImpl(op_name="moe_permute_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.moe_permute_bwd, vendor=None, priority=50),
        OpImpl(op_name="moe_unpermute_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.moe_unpermute_fwd, vendor=None, priority=50),
        OpImpl(op_name="moe_unpermute_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.moe_unpermute_bwd, vendor=None, priority=50),

        # Fused attention
        OpImpl(op_name="get_fused_attn_backend", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.get_fused_attn_backend, vendor=None, priority=50),
        OpImpl(op_name="fused_attn_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_attn_fwd, vendor=None, priority=50),
        OpImpl(op_name="fused_attn_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_attn_bwd, vendor=None, priority=50),
        OpImpl(op_name="fa_prepare_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fa_prepare_fwd, vendor=None, priority=50),
        OpImpl(op_name="fa_prepare_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fa_prepare_bwd, vendor=None, priority=50),

        # KV cache
        OpImpl(op_name="copy_to_kv_cache", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.copy_to_kv_cache, vendor=None, priority=50),

        # Tensor format conversions
        OpImpl(op_name="convert_thd_to_bshd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.convert_thd_to_bshd, vendor=None, priority=50),
        OpImpl(op_name="convert_bshd_to_thd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.convert_bshd_to_thd, vendor=None, priority=50),

        # RoPE (Rotary Position Embedding)
        OpImpl(op_name="fused_rope_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_rope_forward, vendor=None, priority=50),
        OpImpl(op_name="fused_rope_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_rope_backward, vendor=None, priority=50),
        OpImpl(op_name="fused_qkv_rope_forward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_qkv_rope_forward, vendor=None, priority=50),
        OpImpl(op_name="fused_qkv_rope_backward", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_qkv_rope_backward, vendor=None, priority=50),

        # TopK and MOE aux loss
        OpImpl(op_name="fused_topk_with_score_function_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_topk_with_score_function_fwd, vendor=None, priority=50),
        OpImpl(op_name="fused_topk_with_score_function_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_topk_with_score_function_bwd, vendor=None, priority=50),
        OpImpl(op_name="fused_score_for_moe_aux_loss_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_score_for_moe_aux_loss_fwd, vendor=None, priority=50),
        OpImpl(op_name="fused_score_for_moe_aux_loss_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_score_for_moe_aux_loss_bwd, vendor=None, priority=50),
        OpImpl(op_name="fused_moe_aux_loss_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_moe_aux_loss_fwd, vendor=None, priority=50),
        OpImpl(op_name="fused_moe_aux_loss_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_moe_aux_loss_bwd, vendor=None, priority=50),

        # Dropout
        OpImpl(op_name="dropout_fwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dropout_fwd, vendor=None, priority=50),
        OpImpl(op_name="dropout_bwd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.dropout_bwd, vendor=None, priority=50),

        # FP8 operations
        OpImpl(op_name="fp8_transpose", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fp8_transpose, vendor=None, priority=50),
        OpImpl(op_name="swap_first_dims", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.swap_first_dims, vendor=None, priority=50),
        OpImpl(op_name="compute_amax", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.compute_amax, vendor=None, priority=50),
        OpImpl(op_name="fused_amax_and_scale_update_after_reduction", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_amax_and_scale_update_after_reduction, vendor=None, priority=50),
        OpImpl(op_name="fp8_block_scaling_compute_partial_amax", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fp8_block_scaling_compute_partial_amax, vendor=None, priority=50),
        OpImpl(op_name="fp8_block_scaling_partial_cast", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fp8_block_scaling_partial_cast, vendor=None, priority=50),

        # Padding operations
        OpImpl(op_name="fused_multi_row_padding", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_multi_row_padding, vendor=None, priority=50),
        OpImpl(op_name="fused_multi_row_unpadding", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.fused_multi_row_unpadding, vendor=None, priority=50),

        # Library version getters
        OpImpl(op_name="get_cublasLt_version", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.get_cublasLt_version, vendor=None, priority=50),
        OpImpl(op_name="get_cudnn_version", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.get_cudnn_version, vendor=None, priority=50),
        OpImpl(op_name="get_num_cublas_streams", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.get_num_cublas_streams, vendor=None, priority=50),

        # THD (Tensor, Hidden, Dimension) operations
        OpImpl(op_name="thd_read_half_tensor", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.thd_read_half_tensor, vendor=None, priority=50),
        OpImpl(op_name="thd_second_half_lse_correction", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.thd_second_half_lse_correction, vendor=None, priority=50),
        OpImpl(op_name="thd_read_second_half_lse", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.thd_read_second_half_lse, vendor=None, priority=50),
        OpImpl(op_name="thd_out_correction", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.thd_out_correction, vendor=None, priority=50),
        OpImpl(op_name="thd_grad_correction", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.thd_grad_correction, vendor=None, priority=50),
        OpImpl(op_name="thd_get_partitioned_indices", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.thd_get_partitioned_indices, vendor=None, priority=50),

        # NVSHMEM operations
        OpImpl(op_name="init_nvshmem_backend", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.init_nvshmem_backend, vendor=None, priority=50),
        OpImpl(op_name="create_nvshmem_tensor", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.create_nvshmem_tensor, vendor=None, priority=50),
        OpImpl(op_name="nvshmem_send_on_current_stream", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.nvshmem_send_on_current_stream, vendor=None, priority=50),
        OpImpl(op_name="nvshmem_wait_on_current_stream", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.nvshmem_wait_on_current_stream, vendor=None, priority=50),
        OpImpl(op_name="nvshmem_finalize", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.nvshmem_finalize, vendor=None, priority=50),

        # Multi-tensor optimizer operations
        OpImpl(op_name="multi_tensor_scale", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_scale, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_l2norm", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_l2norm, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_unscale_l2norm", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_unscale_l2norm, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_adam", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_adam, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_adam_param_remainder", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_adam_param_remainder, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_adam_fp8", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_adam_fp8, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_adam_capturable", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_adam_capturable, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_adam_capturable_master", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_adam_capturable_master, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_sgd", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_sgd, vendor=None, priority=50),
        OpImpl(op_name="multi_tensor_compute_scale_and_scale_inv", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.multi_tensor_compute_scale_and_scale_inv, vendor=None, priority=50),

        # Communication overlap operations
        OpImpl(op_name="bulk_overlap_ag_with_external_gemm", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.bulk_overlap_ag_with_external_gemm, vendor=None, priority=50),
        OpImpl(op_name="create_fp8_tensor_meta", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.create_fp8_tensor_meta, vendor=None, priority=50),
        OpImpl(op_name="create_comm_overlap_helper", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.create_comm_overlap_helper, vendor=None, priority=50),
        OpImpl(op_name="create_comm_overlap", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.create_comm_overlap, vendor=None, priority=50),
        OpImpl(op_name="create_comm_overlap_p2p", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.create_comm_overlap_p2p, vendor=None, priority=50),

        # FlashAttention class getter
        OpImpl(op_name="get_flash_attention_class", impl_id="reference.torch", kind=BackendImplKind.REFERENCE, fn=backend.get_flash_attention_class, vendor=None, priority=50),
    ]

    registry.register_many(impls)
