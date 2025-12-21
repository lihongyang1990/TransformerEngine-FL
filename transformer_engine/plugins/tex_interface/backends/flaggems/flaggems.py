# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
FlagGems Backend for TEX Interface

FlagGems is a cross-platform backend that can run on multiple hardware.
It has the highest priority and supports fallback to native vendor backends.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Import base classes, registry, and decorators
from ...base import TEXBackendBase, FP8TensorMeta, NVTE_Fused_Attn_Backend
from ...registry import register_backend
from ...decorators import with_fallback, DEBUG
from ...logger import print_once

# Import PyTorch implementations
from .cpp_extensions import (
    rmsnorm_fwd_fl, rmsnorm_bwd_fl,
    multi_tensor_scale_fl, multi_tensor_adam_fl,
    multi_tensor_l2_norm_fl,
    generic_gemm_fl
)


def _check_flaggems_available() -> bool:
    """Check if FlagGems backend is available."""
    # FlagGems is always available as a fallback wrapper
    # It will use native backends underneath
    return True


@register_backend
class FlagGemsBackend(TEXBackendBase):
    """
    FlagGems Cross-Platform Backend.

    This backend provides:
    1. Cross-platform support via FlagGems/Triton
    2. Automatic fallback to native vendor backends
    3. Debug printing for development

    Priority: 150 (highest, to be selected first)
    """

    NAME = "flaggems"
    PRIORITY = 150  # Highest priority

    @staticmethod
    def check_available() -> bool:
        return _check_flaggems_available()

    def __init__(self):
        self._fallback_backend = None
        # Default fallback: try 'torch' first (always available), then 'nvidia'
        self._fallback_name = os.environ.get("TE_FALLBACK_BACKEND", "torch")

    def _get_fallback_backend(self) -> Optional[TEXBackendBase]:
        """Get the fallback backend (lazy initialization)."""
        if self._fallback_backend is None:
            try:
                # Get registry module - try multiple possible module names
                _reg = (
                    sys.modules.get("tex_interface.registry") or
                    sys.modules.get("transformer_engine.plugins.tex_interface.registry")
                )
                if _reg is not None:
                    get_backend = _reg.get_backend
                    list_backends = _reg.list_backends
                else:
                    from ..registry import get_backend, list_backends

                # Try to get the specified fallback backend
                try:
                    self._fallback_backend = get_backend(self._fallback_name)
                    if DEBUG:
                        print_once(f"[FlagGems] Initialized fallback backend: {self._fallback_name}")
                except Exception:
                    # If specified backend not available, try to find another one
                    available = list_backends()
                    # Remove 'flaggems' from candidates to avoid self-reference
                    candidates = [b for b in available if b != "flaggems"]
                    if DEBUG:
                        print_once(f"[FlagGems] Fallback '{self._fallback_name}' not found, available: {candidates}")
                    for candidate in candidates:
                        try:
                            self._fallback_backend = get_backend(candidate)
                            self._fallback_name = candidate
                            if DEBUG:
                                print_once(f"[FlagGems] Using fallback backend: {candidate}")
                            break
                        except Exception:
                            continue
            except Exception as e:
                if DEBUG:
                    print_once(f"[FlagGems] Failed to initialize fallback backend: {e}")
                self._fallback_backend = None
        return self._fallback_backend

    def set_fallback_backend(self, name: str):
        """Set the fallback backend by name."""
        self._fallback_name = name
        self._fallback_backend = None  # Reset to force re-initialization
        if DEBUG:
            print_once(f"[FlagGems] Fallback backend set to: {name}")

    # =========================================================================
    # Backend Metadata
    # =========================================================================

    @property
    def name(self) -> str:
        return "flaggems"

    @property
    def vendor(self) -> str:
        return "FlagGems/BAAI"

    @property
    def priority(self) -> int:
        return 150

    def is_available(self) -> bool:
        return _check_flaggems_available()

    # =========================================================================
    # FlashAttention
    # =========================================================================

    def get_flash_attention_class(self):
        """Return the FlashAttention class for FlagGems backend."""
        from .attention.dot_product_attention.backends import FlashAttentionFL
        return FlashAttentionFL

    # =========================================================================
    # Quantization Operations
    # =========================================================================

    @with_fallback
    def quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        output: Optional[torch.Tensor] = None,
        noop: Optional[torch.Tensor] = None,
    ) -> Any:
        # TODO: Implement FlagGems quantization
        raise NotImplementedError("quantize - implement with FlagGems or fallback")

    @with_fallback
    def dequantize(
        self,
        input: torch.Tensor,
        otype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError("dequantize - implement with FlagGems or fallback")

    @with_fallback
    def bgrad_quantize(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("bgrad_quantize - implement with FlagGems or fallback")

    # =========================================================================
    # GEMM Operations
    # =========================================================================

    @with_fallback
    def generic_gemm(
        self,
        A: torch.Tensor,
        transA: bool,
        B: torch.Tensor,
        transB: bool,
        D: torch.Tensor,
        quantizer: Any,
        output_dtype: torch.dtype,
        bias: Optional[torch.Tensor],
        bias_type: Any,
        gelu: bool,
        gelu_in: Optional[torch.Tensor],
        grad: bool,
        workspace: torch.Tensor,
        workspace_size: int,
        accumulate: bool,
        use_split_accumulator: bool,
        comm_overlap: Optional[Any] = None,
        comm_type: Optional[Any] = None,
        extra_output: Optional[torch.Tensor] = None,
        bulk_overlap: bool = False,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ) -> Any:
        return generic_gemm_fl(
            A,
            transA,
            B,
            transB,
            D,
            quantizer,
            output_dtype,
            bias,
            bias_type,
            gelu,
            gelu_in,
            grad,
            workspace,
            workspace_size,
            accumulate,
            use_split_accumulator,
            comm_overlap=comm_overlap,
            comm_type=comm_type,
            extra_output=extra_output,
            bulk_overlap=bulk_overlap,
            alpha=alpha,
            beta=beta
        )

    @with_fallback
    def te_general_grouped_gemm(self, *args, **kwargs) -> Any:
        raise NotImplementedError("te_general_grouped_gemm - implement with FlagGems or fallback")

    # =========================================================================
    # Activation Functions - Forward
    # =========================================================================

    @with_fallback
    def gelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        # TODO: Implement with FlagGems
        raise NotImplementedError("gelu - implement with FlagGems or fallback")

    @with_fallback
    def geglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("geglu - implement with FlagGems or fallback")

    @with_fallback
    def qgelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("qgelu - implement with FlagGems or fallback")

    @with_fallback
    def qgeglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("qgeglu - implement with FlagGems or fallback")

    @with_fallback
    def relu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("relu - implement with FlagGems or fallback")

    @with_fallback
    def reglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("reglu - implement with FlagGems or fallback")

    @with_fallback
    def srelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("srelu - implement with FlagGems or fallback")

    @with_fallback
    def sreglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("sreglu - implement with FlagGems or fallback")

    @with_fallback
    def silu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("silu - implement with FlagGems or fallback")

    @with_fallback
    def swiglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("swiglu - implement with FlagGems or fallback")

    @with_fallback
    def clamped_swiglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        raise NotImplementedError("clamped_swiglu - implement with FlagGems or fallback")

    # =========================================================================
    # Activation Functions - Backward
    # =========================================================================

    @with_fallback
    def dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dgelu - implement with FlagGems or fallback")

    @with_fallback
    def dgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dgeglu - implement with FlagGems or fallback")

    @with_fallback
    def dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dqgelu - implement with FlagGems or fallback")

    @with_fallback
    def dqgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dqgeglu - implement with FlagGems or fallback")

    @with_fallback
    def drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("drelu - implement with FlagGems or fallback")

    @with_fallback
    def dreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dreglu - implement with FlagGems or fallback")

    @with_fallback
    def dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dsrelu - implement with FlagGems or fallback")

    @with_fallback
    def dsreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dsreglu - implement with FlagGems or fallback")

    @with_fallback
    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dsilu - implement with FlagGems or fallback")

    @with_fallback
    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        raise NotImplementedError("dswiglu - implement with FlagGems or fallback")

    @with_fallback
    def clamped_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        raise NotImplementedError("clamped_dswiglu - implement with FlagGems or fallback")

    # =========================================================================
    # DBias + DAct Fusions
    # =========================================================================

    @with_fallback
    def dbias_dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_dgelu - implement with FlagGems or fallback")

    @with_fallback
    def dbias_dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_dsilu - implement with FlagGems or fallback")

    @with_fallback
    def dbias_drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_drelu - implement with FlagGems or fallback")

    @with_fallback
    def dbias_dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_dqgelu - implement with FlagGems or fallback")

    @with_fallback
    def dbias_dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("dbias_dsrelu - implement with FlagGems or fallback")

    # =========================================================================
    # Normalization Operations
    # =========================================================================

    @with_fallback
    def layernorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Implement with FlagGems
        raise NotImplementedError("layernorm_fwd - implement with FlagGems or fallback")

    @with_fallback
    def layernorm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int = 0,
        zero_centered_gamma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("layernorm_bwd - implement with FlagGems or fallback")

    @with_fallback
    def rmsnorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        return rmsnorm_fwd_fl(
            input=input,
            weight=weight,
            eps=eps,
            ln_out=ln_out,
            quantizer=quantizer,
            odtype=otype,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    @with_fallback
    def rmsnorm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int = 0,
        zero_centered_gamma: bool = False,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rmsnorm_bwd_fl(
            dy=dy,
            x=x,
            rsigma=rsigma,
            gamma=gamma,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
            eps=eps,
        )

    @with_fallback
    def rmsnorm_bwd_add(self, *args, **kwargs) -> Any:
        raise NotImplementedError("rmsnorm_bwd_add - implement with FlagGems or fallback")

    # =========================================================================
    # Multi-tensor Operations
    # =========================================================================

    @with_fallback
    def multi_tensor_quantize(
        self,
        tensor_list: List[torch.Tensor],
        quantizer_list: List[Any],
    ) -> List[Any]:
        raise NotImplementedError("multi_tensor_quantize - implement with FlagGems or fallback")

    @with_fallback
    def split_quantize(
        self,
        tensor: torch.Tensor,
        split_sections: List[int],
        quantizer_list: List[Any],
    ) -> List[Any]:
        raise NotImplementedError("split_quantize - implement with FlagGems or fallback")

    # =========================================================================
    # MOE Permutation Operations
    # =========================================================================

    @with_fallback
    def moe_permute_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_permute_fwd - implement with FlagGems or fallback")

    @with_fallback
    def moe_permute_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_permute_bwd - implement with FlagGems or fallback")

    @with_fallback
    def moe_unpermute_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_unpermute_fwd - implement with FlagGems or fallback")

    @with_fallback
    def moe_unpermute_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_unpermute_bwd - implement with FlagGems or fallback")

    # =========================================================================
    # Softmax Operations
    # =========================================================================

    @with_fallback
    def scaled_softmax_forward(self, input: torch.Tensor, scale: float) -> torch.Tensor:
        raise NotImplementedError("scaled_softmax_forward - implement with FlagGems or fallback")

    @with_fallback
    def scaled_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError("scaled_softmax_backward - implement with FlagGems or fallback")

    @with_fallback
    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError("scaled_masked_softmax_forward - implement with FlagGems or fallback")

    @with_fallback
    def scaled_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError("scaled_masked_softmax_backward - implement with FlagGems or fallback")

    @with_fallback
    def scaled_upper_triang_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError("scaled_upper_triang_masked_softmax_forward - implement with FlagGems or fallback")

    @with_fallback
    def scaled_upper_triang_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError("scaled_upper_triang_masked_softmax_backward - implement with FlagGems or fallback")

    @with_fallback
    def scaled_aligned_causal_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError("scaled_aligned_causal_masked_softmax_forward - implement with FlagGems or fallback")

    @with_fallback
    def scaled_aligned_causal_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError("scaled_aligned_causal_masked_softmax_backward - implement with FlagGems or fallback")

    # =========================================================================
    # Attention Operations
    # =========================================================================

    @with_fallback
    def get_fused_attn_backend(self, *args, **kwargs) -> int:
        return NVTE_Fused_Attn_Backend.NVTE_No_Backend

    @with_fallback
    def fused_attn_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_attn_fwd - implement with FlagGems or fallback")

    @with_fallback
    def fused_attn_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_attn_bwd - implement with FlagGems or fallback")

    @with_fallback
    def fa_prepare_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fa_prepare_fwd - implement with FlagGems or fallback")

    @with_fallback
    def fa_prepare_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fa_prepare_bwd - implement with FlagGems or fallback")

    @with_fallback
    def copy_to_kv_cache(self, *args, **kwargs) -> Any:
        raise NotImplementedError("copy_to_kv_cache - implement with FlagGems or fallback")

    @with_fallback
    def convert_thd_to_bshd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("convert_thd_to_bshd - implement with FlagGems or fallback")

    @with_fallback
    def convert_bshd_to_thd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("convert_bshd_to_thd - implement with FlagGems or fallback")

    # =========================================================================
    # RoPE Operations
    # =========================================================================

    @with_fallback
    def fused_rope_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_rope_forward - implement with FlagGems or fallback")

    @with_fallback
    def fused_rope_backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_rope_backward - implement with FlagGems or fallback")

    @with_fallback
    def fused_qkv_rope_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_qkv_rope_forward - implement with FlagGems or fallback")

    @with_fallback
    def fused_qkv_rope_backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_qkv_rope_backward - implement with FlagGems or fallback")

    # =========================================================================
    # Router Operations (MOE)
    # =========================================================================

    @with_fallback
    def fused_topk_with_score_function_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: int,
        group_topk: int,
        scaling_factor: float,
        score_function: Any,
        expert_bias: Optional[torch.Tensor],
    ) -> Any:
        raise NotImplementedError("fused_topk_with_score_function_fwd - implement with FlagGems or fallback")

    @with_fallback
    def fused_topk_with_score_function_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        routing_map: torch.Tensor,
        intermediate_output: torch.Tensor,
        grad_probs: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        scaling_factor: float,
        score_function: Any,
    ) -> Any:
        raise NotImplementedError("fused_topk_with_score_function_bwd - implement with FlagGems or fallback")

    @with_fallback
    def fused_score_for_moe_aux_loss_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        score_function: Any,
    ) -> Any:
        raise NotImplementedError("fused_score_for_moe_aux_loss_fwd - implement with FlagGems or fallback")

    @with_fallback
    def fused_score_for_moe_aux_loss_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        intermediate_output: torch.Tensor,
        grad_scores: torch.Tensor,
        topk: int,
        score_function: Any,
    ) -> Any:
        raise NotImplementedError("fused_score_for_moe_aux_loss_bwd - implement with FlagGems or fallback")

    @with_fallback
    def fused_moe_aux_loss_fwd(
        self,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: int,
        num_experts: int,
        num_rows: int,
        num_cols: int,
        topk: int,
        coeff: float,
    ) -> Any:
        raise NotImplementedError("fused_moe_aux_loss_fwd - implement with FlagGems or fallback")

    @with_fallback
    def fused_moe_aux_loss_bwd(
        self,
        Const_buf: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_rows: int,
        num_cols: int,
        grad_aux_loss: torch.Tensor,
    ) -> Any:
        raise NotImplementedError("fused_moe_aux_loss_bwd - implement with FlagGems or fallback")

    # =========================================================================
    # Dropout Operations
    # =========================================================================

    @with_fallback
    def dropout_fwd(
        self,
        input: torch.Tensor,
        dropout_probability: float,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("dropout_fwd - implement with FlagGems or fallback")

    @with_fallback
    def dropout_bwd(
        self,
        grad_output: torch.Tensor,
        mask: torch.Tensor,
        dropout_probability: float,
        grad_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("dropout_bwd - implement with FlagGems or fallback")

    # =========================================================================
    # FP8 Utility Operations
    # =========================================================================

    @with_fallback
    def fp8_transpose(
        self,
        input: torch.Tensor,
        dtype: Any,
        *,
        out: torch.Tensor,
    ) -> None:
        raise NotImplementedError("fp8_transpose - implement with FlagGems or fallback")

    @with_fallback
    def swap_first_dims(
        self,
        tensor: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        raise NotImplementedError("swap_first_dims - implement with FlagGems or fallback")

    @with_fallback
    def compute_amax(
        self,
        input: torch.Tensor,
        amax: torch.Tensor,
    ) -> None:
        raise NotImplementedError("compute_amax - implement with FlagGems or fallback")

    @with_fallback
    def fused_amax_and_scale_update_after_reduction(self, *args, **kwargs) -> None:
        raise NotImplementedError("fused_amax_and_scale_update_after_reduction - implement with FlagGems or fallback")

    @with_fallback
    def fp8_block_scaling_compute_partial_amax(
        self,
        tensor: torch.Tensor,
        amax: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
    ) -> None:
        raise NotImplementedError("fp8_block_scaling_compute_partial_amax - implement with FlagGems or fallback")

    @with_fallback
    def fp8_block_scaling_partial_cast(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
        out_dtype: Any,
    ) -> None:
        raise NotImplementedError("fp8_block_scaling_partial_cast - implement with FlagGems or fallback")

    @with_fallback
    def fused_multi_row_padding(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_multi_row_padding - implement with FlagGems or fallback")

    @with_fallback
    def fused_multi_row_unpadding(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_multi_row_unpadding - implement with FlagGems or fallback")

    # =========================================================================
    # Version and Info
    # =========================================================================

    @with_fallback
    def get_cublasLt_version(self) -> int:
        """Return a mock cublasLt version (11.0.0)."""
        return 110000  # Version 11.0.0 (major * 10000 + minor * 100 + patch)

    @with_fallback
    def get_cudnn_version(self) -> int:
        """Return a mock cuDNN version (9.0.0)."""
        return 90000  # Version 9.0.0 (major * 10000 + minor * 100 + patch)

    @with_fallback
    def get_num_cublas_streams(self) -> int:
        """Return number of compute streams."""
        return 0  # No CUDA streams in FL mode

    # =========================================================================
    # Context Parallel (THD format) Operations
    # =========================================================================

    @with_fallback
    def thd_read_half_tensor(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_read_half_tensor - implement with FlagGems or fallback")

    @with_fallback
    def thd_second_half_lse_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_second_half_lse_correction - implement with FlagGems or fallback")

    @with_fallback
    def thd_read_second_half_lse(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_read_second_half_lse - implement with FlagGems or fallback")

    @with_fallback
    def thd_out_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_out_correction - implement with FlagGems or fallback")

    @with_fallback
    def thd_grad_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_grad_correction - implement with FlagGems or fallback")

    @with_fallback
    def thd_get_partitioned_indices(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_get_partitioned_indices - implement with FlagGems or fallback")

    # =========================================================================
    # NVSHMEM Operations
    # =========================================================================

    @with_fallback
    def init_nvshmem_backend(self, *args, **kwargs) -> None:
        raise NotImplementedError("init_nvshmem_backend - implement with FlagGems or fallback")

    @with_fallback
    def create_nvshmem_tensor(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("create_nvshmem_tensor - implement with FlagGems or fallback")

    @with_fallback
    def nvshmem_send_on_current_stream(self, *args, **kwargs) -> None:
        raise NotImplementedError("nvshmem_send_on_current_stream - implement with FlagGems or fallback")

    @with_fallback
    def nvshmem_wait_on_current_stream(self, *args, **kwargs) -> None:
        raise NotImplementedError("nvshmem_wait_on_current_stream - implement with FlagGems or fallback")

    @with_fallback
    def nvshmem_finalize(self) -> None:
        raise NotImplementedError("nvshmem_finalize - implement with FlagGems or fallback")

    # =========================================================================
    # Multi-tensor Optimizer Operations
    # =========================================================================

    @with_fallback
    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        return multi_tensor_scale_fl(chunk_size, noop_flag, tensor_lists, scale)

    @with_fallback
    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        result, _ = multi_tensor_l2_norm_fl(chunk_size, noop_flag, tensor_lists, per_tensor)
        return result

    @with_fallback
    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: torch.Tensor,
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError("multi_tensor_unscale_l2norm - not yet implemented in FlagGems")

    @with_fallback
    def multi_tensor_adam(
        self,
        chunk_size: int = None,
        noop_flag: torch.Tensor = None,
        tensor_lists: List[List[torch.Tensor]] = None,
        lr: float = None,
        beta1: float = None,
        beta2: float = None,
        eps: float = None,
        step: int = None,
        mode: int = None,
        bias_correction: int = None,
        weight_decay: float = None,
    ):
        # If called without arguments, return the callable function itself
        # This matches NVIDIA tex behavior where tex.multi_tensor_adam() returns the function
        if chunk_size is None:
            return multi_tensor_adam_fl
        return multi_tensor_adam_fl(
            chunk_size=chunk_size,
            noop_flag=noop_flag,
            tensor_lists=tensor_lists,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            step=step,
            mode=mode,
            bias_correction=bias_correction,
            weight_decay=weight_decay,
        )

    @with_fallback
    def multi_tensor_adam_param_remainder(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_param_remainder - implement with FlagGems or fallback")

    @with_fallback
    def multi_tensor_adam_fp8(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_fp8 - implement with FlagGems or fallback")

    @with_fallback
    def multi_tensor_adam_capturable(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_capturable - implement with FlagGems or fallback")

    @with_fallback
    def multi_tensor_adam_capturable_master(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_capturable_master - implement with FlagGems or fallback")

    @with_fallback
    def multi_tensor_sgd(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_sgd - implement with FlagGems or fallback")

    @with_fallback
    def multi_tensor_compute_scale_and_scale_inv(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_compute_scale_and_scale_inv - implement with FlagGems or fallback")

    # =========================================================================
    # Communication Overlap
    # =========================================================================

    @with_fallback
    def bulk_overlap_ag_with_external_gemm(
        self,
        allgather_communicator: Any,
        send_stream: Any,
        recv_stream: Any,
    ) -> Any:
        raise NotImplementedError("bulk_overlap_ag_with_external_gemm - implement with FlagGems or fallback")

    # =========================================================================
    # Data Structure Creation
    # =========================================================================

    @with_fallback
    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        return FP8TensorMeta()

    @with_fallback
    def create_comm_overlap_helper(
        self,
        world_group: Optional[Any] = None,
        intra_node_group: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError("create_comm_overlap_helper - implement with FlagGems or fallback")

    @with_fallback
    def create_comm_overlap(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        num_splits: int = 3,
        num_max_streams: int = 3,
        comm_cga_size: int = 2,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 16,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
        rs_overlap_first_gemm: bool = False,
    ) -> Any:
        raise NotImplementedError("create_comm_overlap - implement with FlagGems or fallback")

    @with_fallback
    def create_comm_overlap_p2p(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        comm_type: Any,
        num_max_streams: int = 3,
        comm_cga_size: int = 1,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 1,
        set_sm_margin: bool = False,
        atomic_gemm: bool = False,
        use_ce: bool = True,
        aggregate: bool = False,
    ) -> Any:
        raise NotImplementedError("create_comm_overlap_p2p - implement with FlagGems or fallback")
