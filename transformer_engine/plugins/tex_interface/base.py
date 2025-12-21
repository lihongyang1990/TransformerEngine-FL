# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Abstract Base Class for Transformer Engine Torch Interface

This module defines all the interfaces that match transformer_engine_torch (tex) pybind.cpp.
Vendors should inherit from TEXBackendBase and implement all methods.

The interface is designed to be a drop-in replacement:
    # Original:
    import transformer_engine_torch as tex
    tex.rmsnorm_fwd(...)

    # With this interface:
    import transformer_engine_fl_torch as tex  # tex is a TEXModule instance
    tex.rmsnorm_fwd(...)  # Exactly the same API
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from enum import IntEnum

import torch


# =============================================================================
# Enum Types (Must match C++ definitions in transformer_engine_torch)
# =============================================================================
class DType(IntEnum):
    """Mock TE DType enum matching transformer_engine_torch.DType

    Values must match the C++ enum in transformer_engine_torch_nv:
    - kByte = 0
    - kInt32 = 2
    - kFloat32 = 4
    - kFloat16 = 5
    - kBFloat16 = 6
    - kFloat8E4M3 = 7
    - kFloat8E5M2 = 8
    - kFloat4E2M1 = 10
    """
    kByte = 0
    kInt32 = 2
    kFloat32 = 4
    kFloat16 = 5
    kBFloat16 = 6
    kFloat8E4M3 = 7
    kFloat8E5M2 = 8
    kFloat4E2M1 = 10

class Float8BlockScaleTensorFormat(IntEnum):
    """Float8 block scale tensor format."""
    COMPACT = 0
    GEMM_READY = 1

class NVTE_Activation_Type(IntEnum):
    """Activation type for fused operations."""
    NVTE_GELU = 0
    NVTE_GEGLU = 1
    NVTE_SILU = 2
    NVTE_SWIGLU = 3
    NVTE_RELU = 4
    NVTE_REGLU = 5
    NVTE_QGELU = 6
    NVTE_QGEGLU = 7
    NVTE_SRELU = 8
    NVTE_SREGLU = 9

class NVTE_Softmax_Type(IntEnum):
    """Softmax type."""
    NVTE_VANILLA_SOFTMAX = 0
    NVTE_OFF_BY_ONE_SOFTMAX = 1
    NVTE_LEARNABLE_SOFTMAX = 2

class CommGemmOverlapRole(IntEnum):
    """Communication GEMM overlap role."""
    INPUT = 0
    OUTPUT = 1

class FP8FwdTensors(IntEnum):
    """FP8 forward tensor indices."""
    GEMM1_INPUT = 0
    GEMM1_WEIGHT = 1
    GEMM1_OUTPUT = 2
    GEMM2_INPUT = 3
    GEMM2_WEIGHT = 4
    GEMM2_OUTPUT = 5
    GEMM3_INPUT = 6
    GEMM3_WEIGHT = 7
    GEMM3_OUTPUT = 8

class FP8BwdTensors(IntEnum):
    """FP8 backward tensor indices."""
    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
    GRAD_OUTPUT2 = 2
    GRAD_INPUT2 = 3
    GRAD_OUTPUT3 = 4
    GRAD_INPUT3 = 5

class NVTE_Bias_Type(IntEnum):
    """Bias type for attention."""
    NVTE_NO_BIAS = 0
    NVTE_PRE_SCALE_BIAS = 1
    NVTE_POST_SCALE_BIAS = 2
    NVTE_ALIBI = 3

class NVTE_Mask_Type(IntEnum):
    """Mask type for attention."""
    NVTE_NO_MASK = 0
    NVTE_PADDING_MASK = 1
    NVTE_CAUSAL_MASK = 2
    NVTE_PADDING_CAUSAL_MASK = 3
    NVTE_CAUSAL_BOTTOM_RIGHT_MASK = 4
    NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK = 5
    NVTE_ARBITRARY_MASK = 6

class NVTE_Fused_Attn_Backend(IntEnum):
    """Fused attention backend type."""
    NVTE_No_Backend = 0
    NVTE_F16_max512_seqlen = 1
    NVTE_F16_arbitrary_seqlen = 2
    NVTE_FP8 = 3
    NVTE_FA3 = 4

class NVTE_QKV_Format(IntEnum):
    """QKV tensor format."""
    NVTE_BSHD = 0
    NVTE_SBHD = 1
    NVTE_THD = 2
    NVTE_SBHD_2BSHD = 3
    NVTE_BSHD_2SBHD = 4
    NVTE_THD_2BSHD = 5
    NVTE_THD_2SBHD = 6

class NVTE_QKV_Layout(IntEnum):
    """QKV tensor layout."""
    NVTE_SB3HD = 0
    NVTE_SBH3D = 1
    NVTE_SBHD_SB2HD = 2
    NVTE_SBHD_SBH2D = 3
    NVTE_SBHD_SBHD_SBHD = 4
    NVTE_BS3HD = 5
    NVTE_BSH3D = 6
    NVTE_BSHD_BS2HD = 7
    NVTE_BSHD_BSH2D = 8
    NVTE_BSHD_BSHD_BSHD = 9
    NVTE_T3HD = 10
    NVTE_TH3D = 11
    NVTE_THD_T2HD = 12
    NVTE_THD_TH2D = 13
    NVTE_THD_THD_THD = 14
    NVTE_SBHD_BSHD_BSHD = 15
    NVTE_BSHD_SBHD_SBHD = 16
    NVTE_THD_BSHD_BSHD = 17
    NVTE_THD_SBHD_SBHD = 18
    NVTE_Paged_KV_BSHD_BSHD_BSHD = 19
    NVTE_Paged_KV_BSHD_SBHD_SBHD = 20
    NVTE_Paged_KV_SBHD_BSHD_BSHD = 21
    NVTE_Paged_KV_SBHD_SBHD_SBHD = 22
    NVTE_Paged_KV_THD_BSHD_BSHD = 23
    NVTE_Paged_KV_THD_SBHD_SBHD = 24

class CommOverlapType(IntEnum):
    """Communication overlap type."""
    RS = 0
    AG = 1

class CommOverlapAlgo(IntEnum):
    """Communication overlap algorithm."""
    BULK_OVERLAP_AG = 0
    BULK_OVERLAP_RS = 1
    SPLIT_PIPELINED_AG_P2P = 2
    SPLIT_PIPELINED_RS = 3
    SPLIT_PIPELINED_RS_P2P = 4
    ATOMIC_GEMM_RS = 5
    ATOMIC_GEMM_AG_P2P = 6
    ATOMIC_GEMM_RS_P2P = 7
    EXTERNAL_BULK_OVERLAP_AG = 8


# =============================================================================
# Data Classes
# =============================================================================

class FP8TensorMeta:
    """FP8 tensor metadata - matches tex.FP8TensorMeta."""
    def __init__(self):
        self.scale: Optional[torch.Tensor] = None
        self.scale_inv: Optional[torch.Tensor] = None
        self.amax_history: Optional[torch.Tensor] = None


class CommGemmOverlapAlgoConfig:
    """Communication GEMM overlap algorithm config - placeholder."""
    def __init__(self, *args, **kwargs):
        pass


class FusedAdamCUDAKernel:
    """Fused Adam CUDA kernel - placeholder."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FusedAdamCUDAKernel requires CUDA extensions. "
            "Not supported in FL mode."
        )


class FusedSGDCUDAKernel:
    """Fused SGD CUDA kernel - placeholder."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FusedSGDCUDAKernel requires CUDA extensions. "
            "Not supported in FL mode."
        )


class CommOverlapHelper:
    """
    Communication overlap helper class - placeholder.

    This class should be created via backend.create_comm_overlap_helper().
    For compatibility, we provide a minimal implementation that delegates to the backend.
    """
    def __init__(self, world_group=None, intra_node_group=None):
        # This is a placeholder - actual implementation should come from backend
        self.world_group = world_group
        self.intra_node_group = intra_node_group


class CommOverlap:
    """
    Communication overlap class - placeholder.

    This class should be created via backend.create_comm_overlap().
    For compatibility, we provide a minimal implementation.
    """
    def __init__(self, *args, **kwargs):
        # This is a placeholder - actual implementation should come from backend
        raise NotImplementedError(
            "CommOverlap should be created via backend.create_comm_overlap(). "
            "Direct instantiation is not supported in FL mode."
        )


class CommOverlapP2P:
    """
    Point-to-point communication overlap class - placeholder.

    This class should be created via backend.create_comm_overlap_p2p().
    For compatibility, we provide a minimal implementation.
    """
    def __init__(self, *args, **kwargs):
        # This is a placeholder - actual implementation should come from backend
        raise NotImplementedError(
            "CommOverlapP2P should be created via backend.create_comm_overlap_p2p(). "
            "Direct instantiation is not supported in FL mode."
        )


# =============================================================================
# Abstract Backend Base Class
# =============================================================================

class TEXBackendBase(ABC):
    """
    Abstract base class defining all transformer_engine_torch interfaces.

    This class mirrors the complete pybind.cpp interface. Vendors must implement
    all abstract methods. Methods with default implementations can be overridden
    for optimization.

    Interface Categories:
    1. Quantization: quantize, dequantize, bgrad_quantize
    2. GEMM: generic_gemm, te_general_grouped_gemm
    3. Activations: gelu, silu, swiglu, relu, etc. and their backwards
    4. Normalization: layernorm_fwd/bwd, rmsnorm_fwd/bwd
    5. Softmax: scaled_softmax_*, scaled_masked_softmax_*, etc.
    6. Attention: fused_attn_fwd/bwd, fa_prepare_fwd/bwd
    7. MOE: moe_permute_*, moe_unpermute_*
    8. Optimizers: multi_tensor_adam, multi_tensor_sgd, etc.
    9. Communication: CommOverlap, CommOverlapP2P, nvshmem_*
    10. Utilities: fp8_transpose, compute_amax, fused_rope_*, etc.
    """

    # =========================================================================
    # Backend Metadata
    # =========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'nvidia', 'hygon', 'amd')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Vendor name for display."""
        raise NotImplementedError

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority for auto-selection (higher = preferred)."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        raise NotImplementedError

    # =========================================================================
    # FlashAttention
    # =========================================================================

    @abstractmethod
    def get_flash_attention_class(self) -> Type["FlashAttentionBase"]:
        """
        Return the FlashAttention class for this backend.

        Returns:
            A FlashAttentionBase subclass (not instance!)

        Example:
            class MyBackend(TEXBackendBase):
                def get_flash_attention_class(self):
                    return FlashAttentionMyBackend  # Return class, not instance
        """
        raise NotImplementedError

    # =========================================================================
    # Quantization Operations
    # =========================================================================

    @abstractmethod
    def quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        output: Optional[torch.Tensor] = None,
        noop: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Quantize tensor to FP8 format.

        Args:
            tensor: Input tensor to quantize
            quantizer: Quantizer object (Float8Quantizer, etc.)
            output: Optional pre-allocated output
            noop: Optional noop flag tensor

        Returns:
            Quantized tensor or QuantizedTensor object
        """
        raise NotImplementedError

    @abstractmethod
    def dequantize(
        self,
        input: torch.Tensor,
        otype: torch.dtype,
    ) -> torch.Tensor:
        """
        Dequantize FP8 tensor back to higher precision.

        Args:
            input: FP8 quantized tensor
            otype: Output data type

        Returns:
            Dequantized tensor
        """
        raise NotImplementedError

    @abstractmethod
    def bgrad_quantize(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Compute bias gradient and quantize.

        Args:
            input: Input tensor
            quantizer: Quantizer object

        Returns:
            Tuple of (bias_grad, quantized_output)
        """
        raise NotImplementedError

    # =========================================================================
    # GEMM Operations
    # =========================================================================

    @abstractmethod
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
        """
        General matrix multiplication with optional fusions.

        This is the main GEMM interface supporting:
        - FP8/BF16/FP16 inputs
        - Fused bias addition
        - Fused GELU activation
        - Communication overlap for distributed training

        Args:
            A: First input matrix
            transA: Whether to transpose A
            B: Second input matrix
            transB: Whether to transpose B
            D: Output matrix
            quantizer: Output quantizer (can be None)
            output_dtype: Output data type
            bias: Optional bias tensor
            bias_type: Type of bias (NO_BIAS, PRE_SCALE_BIAS, etc.)
            gelu: Whether to apply GELU
            gelu_in: Pre-computed GELU input (for backward)
            grad: Whether this is gradient computation
            workspace: Workspace tensor
            workspace_size: Size of workspace
            accumulate: Whether to accumulate into D
            use_split_accumulator: Use split accumulator for FP8
            comm_overlap: Communication overlap handle
            comm_type: Communication overlap type
            extra_output: Extra output tensor
            bulk_overlap: Enable bulk communication overlap
            alpha: Scaling factor
            beta: Accumulation scaling factor

        Returns:
            Result tensor or tuple of results
        """
        raise NotImplementedError

    @abstractmethod
    def te_general_grouped_gemm(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """
        Grouped GEMM for MoE workloads.

        See generic_gemm for parameter descriptions.
        """
        raise NotImplementedError

    # =========================================================================
    # Activation Functions - Forward
    # =========================================================================

    @abstractmethod
    def gelu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """GeLU activation with optional quantization."""
        raise NotImplementedError

    @abstractmethod
    def geglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """GeGLU activation."""
        raise NotImplementedError

    @abstractmethod
    def qgelu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """QuickGELU activation."""
        raise NotImplementedError

    @abstractmethod
    def qgeglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """QuickGeGLU activation."""
        raise NotImplementedError

    @abstractmethod
    def relu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """ReLU activation."""
        raise NotImplementedError

    @abstractmethod
    def reglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """ReGLU activation."""
        raise NotImplementedError

    @abstractmethod
    def srelu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Squared ReLU activation."""
        raise NotImplementedError

    @abstractmethod
    def sreglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Squared ReGLU activation."""
        raise NotImplementedError

    @abstractmethod
    def silu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """SiLU activation."""
        raise NotImplementedError

    @abstractmethod
    def swiglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """SwiGLU activation."""
        raise NotImplementedError

    @abstractmethod
    def clamped_swiglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        """SwiGLU activation used in GPT OSS."""
        raise NotImplementedError

    # =========================================================================
    # Activation Functions - Backward
    # =========================================================================

    @abstractmethod
    def dgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of GeLU."""
        raise NotImplementedError

    @abstractmethod
    def dgeglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of GeGLU."""
        raise NotImplementedError

    @abstractmethod
    def dqgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of QuickGELU."""
        raise NotImplementedError

    @abstractmethod
    def dqgeglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of QuickGeGLU."""
        raise NotImplementedError

    @abstractmethod
    def drelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of ReLU."""
        raise NotImplementedError

    @abstractmethod
    def dreglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of ReGLU."""
        raise NotImplementedError

    @abstractmethod
    def dsrelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of Squared ReLU."""
        raise NotImplementedError

    @abstractmethod
    def dsreglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of Squared ReGLU."""
        raise NotImplementedError

    @abstractmethod
    def dsilu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of SiLU."""
        raise NotImplementedError

    @abstractmethod
    def dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        """Backward of SwiGLU."""
        raise NotImplementedError

    @abstractmethod
    def clamped_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        """Backward of SwiGLU used in GPT OSS."""
        raise NotImplementedError

    # =========================================================================
    # DBias + DAct Fusions
    # =========================================================================

    @abstractmethod
    def dbias_dgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """DGeLU + DBias + Quantize."""
        raise NotImplementedError

    @abstractmethod
    def dbias_dsilu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """DSiLU + DBias + Quantize."""
        raise NotImplementedError

    @abstractmethod
    def dbias_drelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """DReLU + DBias + Quantize."""
        raise NotImplementedError

    @abstractmethod
    def dbias_dqgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """DQGeLU + DBias + Quantize."""
        raise NotImplementedError

    @abstractmethod
    def dbias_dsrelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """DSquaredReLU + DBias + Quantize."""
        raise NotImplementedError

    # =========================================================================
    # Normalization Operations
    # =========================================================================

    @abstractmethod
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
        """
        LayerNorm forward pass.

        Returns:
            Tuple of (output, mean, rsigma)
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        LayerNorm backward pass.

        Returns:
            Tuple of (dx, dgamma, dbeta)
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        RMSNorm forward pass.

        Returns:
            Tuple of (output, None, rsigma) - None for compatibility with LayerNorm
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        RMSNorm backward pass.

        Args:
            dy: Gradient of output
            x: Input tensor from forward
            rsigma: Inverse RMS from forward
            gamma: Weight tensor
            sm_margin: SM margin for kernel launch
            zero_centered_gamma: Whether gamma is zero-centered
            eps: Epsilon for numerical stability (used in PyTorch implementation)

        Returns:
            Tuple of (dx, dgamma)
        """
        raise NotImplementedError

    @abstractmethod
    def rmsnorm_bwd_add(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused backward of RMSNorm + add."""
        raise NotImplementedError

    # =========================================================================
    # Multi-tensor Operations
    # =========================================================================

    @abstractmethod
    def multi_tensor_quantize(
        self,
        tensor_list: List[torch.Tensor],
        quantizer_list: List[Any],
    ) -> List[Any]:
        """Multi-tensor quantize."""
        raise NotImplementedError

    @abstractmethod
    def split_quantize(
        self,
        tensor: torch.Tensor,
        split_sections: List[int],
        quantizer_list: List[Any],
    ) -> List[Any]:
        """Split and multi-tensor quantize."""
        raise NotImplementedError

    # =========================================================================
    # MOE Permutation Operations
    # =========================================================================

    @abstractmethod
    def moe_permute_fwd(self, *args, **kwargs) -> Any:
        """MOE permute forward."""
        raise NotImplementedError

    @abstractmethod
    def moe_permute_bwd(self, *args, **kwargs) -> Any:
        """MOE permute backward."""
        raise NotImplementedError

    @abstractmethod
    def moe_unpermute_fwd(self, *args, **kwargs) -> Any:
        """MOE unpermute forward."""
        raise NotImplementedError

    @abstractmethod
    def moe_unpermute_bwd(self, *args, **kwargs) -> Any:
        """MOE unpermute backward."""
        raise NotImplementedError

    # =========================================================================
    # Softmax Operations
    # =========================================================================

    @abstractmethod
    def scaled_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scaled softmax forward."""
        raise NotImplementedError

    @abstractmethod
    def scaled_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scaled softmax backward."""
        raise NotImplementedError

    @abstractmethod
    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scaled masked softmax forward."""
        raise NotImplementedError

    @abstractmethod
    def scaled_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scaled masked softmax backward."""
        raise NotImplementedError

    @abstractmethod
    def scaled_upper_triang_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scaled upper-triangular masked softmax forward."""
        raise NotImplementedError

    @abstractmethod
    def scaled_upper_triang_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scaled upper-triangular masked softmax backward."""
        raise NotImplementedError

    @abstractmethod
    def scaled_aligned_causal_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scaled bottom-right corner aligned masked softmax forward."""
        raise NotImplementedError

    @abstractmethod
    def scaled_aligned_causal_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scaled bottom-right corner aligned masked softmax backward."""
        raise NotImplementedError

    # =========================================================================
    # Attention Operations
    # =========================================================================

    @abstractmethod
    def get_fused_attn_backend(
        self,
        *args,
        **kwargs,
    ) -> int:
        """Get fused attention backend type."""
        raise NotImplementedError

    @abstractmethod
    def fused_attn_fwd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused attention forward with separate Q, K, V."""
        raise NotImplementedError

    @abstractmethod
    def fused_attn_bwd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused attention backward with separate Q, K, V."""
        raise NotImplementedError

    @abstractmethod
    def fa_prepare_fwd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Prepare QKV for Flash Attention."""
        raise NotImplementedError

    @abstractmethod
    def fa_prepare_bwd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Backward of QKV preparation for Flash Attention."""
        raise NotImplementedError

    @abstractmethod
    def copy_to_kv_cache(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Copy new KV tokens to KV cache."""
        raise NotImplementedError

    @abstractmethod
    def convert_thd_to_bshd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Convert tensor from THD to BSHD format."""
        raise NotImplementedError

    @abstractmethod
    def convert_bshd_to_thd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Convert tensor from BSHD to THD format."""
        raise NotImplementedError

    # =========================================================================
    # RoPE Operations
    # =========================================================================

    @abstractmethod
    def fused_rope_forward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused Apply RoPE forward."""
        raise NotImplementedError

    @abstractmethod
    def fused_rope_backward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused Apply RoPE backward."""
        raise NotImplementedError

    @abstractmethod
    def fused_qkv_rope_forward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused Apply QKV RoPE forward."""
        raise NotImplementedError

    @abstractmethod
    def fused_qkv_rope_backward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused Apply QKV RoPE backward."""
        raise NotImplementedError

    # =========================================================================
    # Router Operations (MOE)
    # =========================================================================

    @abstractmethod
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
        """Fused topk softmax forward."""
        raise NotImplementedError

    @abstractmethod
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
        """Fused topk softmax backward."""
        raise NotImplementedError

    @abstractmethod
    def fused_score_for_moe_aux_loss_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        score_function: Any,
    ) -> Any:
        """Fused score for MOE aux loss forward."""
        raise NotImplementedError

    @abstractmethod
    def fused_score_for_moe_aux_loss_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        intermediate_output: torch.Tensor,
        grad_scores: torch.Tensor,
        topk: int,
        score_function: Any,
    ) -> Any:
        """Fused score for MOE aux loss backward."""
        raise NotImplementedError

    @abstractmethod
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
        """Fused aux loss forward."""
        raise NotImplementedError

    @abstractmethod
    def fused_moe_aux_loss_bwd(
        self,
        Const_buf: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_rows: int,
        num_cols: int,
        grad_aux_loss: torch.Tensor,
    ) -> Any:
        """Fused aux loss backward."""
        raise NotImplementedError

    # =========================================================================
    # Dropout Operations
    # =========================================================================

    @abstractmethod
    def dropout_fwd(
        self,
        input: torch.Tensor,
        dropout_probability: float,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dropout forward with 8-bit RNG.

        Returns:
            Tuple of (output, mask)
        """
        raise NotImplementedError

    @abstractmethod
    def dropout_bwd(
        self,
        grad_output: torch.Tensor,
        mask: torch.Tensor,
        dropout_probability: float,
        grad_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dropout backward with 8-bit RNG."""
        raise NotImplementedError

    # =========================================================================
    # FP8 Utility Operations
    # =========================================================================

    @abstractmethod
    def fp8_transpose(
        self,
        input: torch.Tensor,
        dtype: Any,
        *,
        out: torch.Tensor,
    ) -> None:
        """Transpose with FP8 I/O."""
        raise NotImplementedError

    @abstractmethod
    def swap_first_dims(
        self,
        tensor: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        """Swap first two tensor dimensions."""
        raise NotImplementedError

    @abstractmethod
    def compute_amax(
        self,
        input: torch.Tensor,
        amax: torch.Tensor,
    ) -> None:
        """Compute absolute max value in tensor."""
        raise NotImplementedError

    @abstractmethod
    def fused_amax_and_scale_update_after_reduction(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Update amax history and FP8 scale/scale_inv after reduction."""
        raise NotImplementedError

    @abstractmethod
    def fp8_block_scaling_compute_partial_amax(
        self,
        tensor: torch.Tensor,
        amax: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
    ) -> None:
        """Compute partial amax from master weights for fp8 block scaling."""
        raise NotImplementedError

    @abstractmethod
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
        """Partial cast from master weights for fp8 block scaling."""
        raise NotImplementedError

    @abstractmethod
    def fused_multi_row_padding(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused multi-tensor padding."""
        raise NotImplementedError

    @abstractmethod
    def fused_multi_row_unpadding(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Fused multi-tensor unpadding."""
        raise NotImplementedError

    # =========================================================================
    # Version and Info
    # =========================================================================

    @abstractmethod
    def get_cublasLt_version(self) -> int:
        """Get cublasLt version (or equivalent library version)."""
        raise NotImplementedError

    @abstractmethod
    def get_cudnn_version(self) -> int:
        """Get cuDNN version (or equivalent library version)."""
        raise NotImplementedError

    @abstractmethod
    def get_num_cublas_streams(self) -> int:
        """Get number of compute streams."""
        raise NotImplementedError

    # =========================================================================
    # Context Parallel (THD format) Operations
    # =========================================================================

    @abstractmethod
    def thd_read_half_tensor(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Read half of THD tensor."""
        raise NotImplementedError

    @abstractmethod
    def thd_second_half_lse_correction(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Correct second half of softmax_lse."""
        raise NotImplementedError

    @abstractmethod
    def thd_read_second_half_lse(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Read second half of softmax_lse."""
        raise NotImplementedError

    @abstractmethod
    def thd_out_correction(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Correct THD format output in forward pass."""
        raise NotImplementedError

    @abstractmethod
    def thd_grad_correction(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Correct THD format gradients in backward pass."""
        raise NotImplementedError

    @abstractmethod
    def thd_get_partitioned_indices(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Generate partitioned indices for THD inputs."""
        raise NotImplementedError

    # =========================================================================
    # NVSHMEM Operations
    # =========================================================================

    @abstractmethod
    def init_nvshmem_backend(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize nvshmem backend with distributed process groups."""
        raise NotImplementedError

    @abstractmethod
    def create_nvshmem_tensor(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Create tensor in NVSHMEM shared memory."""
        raise NotImplementedError

    @abstractmethod
    def nvshmem_send_on_current_stream(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Send tensor data to remote PE using NVSHMEM."""
        raise NotImplementedError

    @abstractmethod
    def nvshmem_wait_on_current_stream(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Wait for signal from remote PE using NVSHMEM."""
        raise NotImplementedError

    @abstractmethod
    def nvshmem_finalize(self) -> None:
        """Clean up NVSHMEM backend."""
        raise NotImplementedError

    # =========================================================================
    # Multi-tensor Optimizer Operations
    # =========================================================================

    @abstractmethod
    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        """Fused overflow check + scale for list of tensors."""
        raise NotImplementedError

    @abstractmethod
    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute L2 norm for list of tensors."""
        raise NotImplementedError

    @abstractmethod
    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: torch.Tensor,
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute L2 norm after unscaling."""
        raise NotImplementedError

    @abstractmethod
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
        """Fused multi-tensor Adam optimizer step.

        When called without arguments, returns the underlying callable function.
        This matches the usage pattern in fused_adam.py:
            self.multi_tensor_adam = tex.multi_tensor_adam
            apply_multi_tensor_adam(self.multi_tensor_adam(), tensor_lists)
        """
        raise NotImplementedError

    @abstractmethod
    def multi_tensor_adam_param_remainder(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Adam optimizer with remainder bits for master params."""
        raise NotImplementedError

    @abstractmethod
    def multi_tensor_adam_fp8(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Adam optimizer for FP8 parameters."""
        raise NotImplementedError

    @abstractmethod
    def multi_tensor_adam_capturable(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Adam optimizer with CUDA graph support."""
        raise NotImplementedError

    @abstractmethod
    def multi_tensor_adam_capturable_master(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Adam optimizer with CUDA graph support and master weights."""
        raise NotImplementedError

    @abstractmethod
    def multi_tensor_sgd(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Fused SGD optimizer for list of tensors."""
        raise NotImplementedError

    @abstractmethod
    def multi_tensor_compute_scale_and_scale_inv(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Compute scale and scale_inv from amax."""
        raise NotImplementedError

    # =========================================================================
    # Communication Overlap
    # =========================================================================

    @abstractmethod
    def bulk_overlap_ag_with_external_gemm(
        self,
        allgather_communicator: Any,
        send_stream: Any,
        recv_stream: Any,
    ) -> Any:
        """Bulk overlap All-Gather with external GEMM."""
        raise NotImplementedError

    # =========================================================================
    # Data Structure Classes (to be created by get_* methods)
    # =========================================================================

    @abstractmethod
    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        """Create FP8TensorMeta instance."""
        raise NotImplementedError

    @abstractmethod
    def create_comm_overlap_helper(
        self,
        world_group: Optional[Any] = None,
        intra_node_group: Optional[Any] = None,
    ) -> Any:
        """Create CommOverlapHelper instance."""
        raise NotImplementedError

    @abstractmethod
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
        """Create CommOverlap instance."""
        raise NotImplementedError

    @abstractmethod
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
        """Create CommOverlapP2P instance."""
        raise NotImplementedError


# =============================================================================
# TEXModule - The module that mimics transformer_engine_torch
# =============================================================================

class TEXModule:
    """
    A module class that wraps TEXBackendBase and exposes all methods as module attributes.

    This allows usage like:
        import transformer_engine_fl_torch as tex
        tex.rmsnorm_fwd(...)

    Instead of:
        tex.backend.rmsnorm_fwd(...)
    """

    def __init__(self, backend: TEXBackendBase, registry_funcs: Optional[Dict[str, Callable]] = None):
        self._backend = backend
        self._registry_funcs = registry_funcs or {}

        # Expose enums and data classes
        self.DType = DType
        self.Float8BlockScaleTensorFormat = Float8BlockScaleTensorFormat
        self.FP8FwdTensors = FP8FwdTensors
        self.FP8BwdTensors = FP8BwdTensors
        self.FP8TensorMeta = FP8TensorMeta
        self.NVTE_Activation_Type = NVTE_Activation_Type
        self.NVTE_Bias_Type = NVTE_Bias_Type
        self.NVTE_Mask_Type = NVTE_Mask_Type
        self.NVTE_Softmax_Type = NVTE_Softmax_Type
        self.NVTE_Fused_Attn_Backend = NVTE_Fused_Attn_Backend
        self.NVTE_QKV_Format = NVTE_QKV_Format
        self.NVTE_QKV_Layout = NVTE_QKV_Layout
        self.CommOverlapType = CommOverlapType
        self.CommOverlapAlgo = CommOverlapAlgo
        self.CommGemmOverlapRole = CommGemmOverlapRole

        # Expose communication overlap classes
        self.CommOverlapHelper = CommOverlapHelper
        self.CommOverlap = CommOverlap
        self.CommOverlapP2P = CommOverlapP2P
        self.CommGemmOverlapAlgoConfig = CommGemmOverlapAlgoConfig

        # Expose optimizer kernel classes
        self.FusedAdamCUDAKernel = FusedAdamCUDAKernel
        self.FusedSGDCUDAKernel = FusedSGDCUDAKernel

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to backend or registry functions."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if it's a registry function
        if name in self._registry_funcs:
            return self._registry_funcs[name]

        # Otherwise delegate to backend
        return getattr(self._backend, name)

    def __dir__(self):
        """Return list of available attributes for dir() and import *."""
        # Get all public attributes from backend
        backend_attrs = [attr for attr in dir(self._backend) if not attr.startswith('_')]

        # Get all module-level attributes
        module_attrs = [
            'DType', 'Float8BlockScaleTensorFormat', 'FP8FwdTensors', 'FP8BwdTensors',
            'FP8TensorMeta', 'NVTE_Activation_Type', 'NVTE_Bias_Type', 'NVTE_Mask_Type',
            'NVTE_Softmax_Type', 'NVTE_Fused_Attn_Backend', 'NVTE_QKV_Format', 'NVTE_QKV_Layout',
            'CommOverlapType', 'CommOverlapAlgo', 'CommGemmOverlapRole',
            'CommOverlapHelper', 'CommOverlap', 'CommOverlapP2P', 'CommGemmOverlapAlgoConfig',
            'FusedAdamCUDAKernel', 'FusedSGDCUDAKernel'
        ]

        # Get registry functions
        registry_attrs = list(self._registry_funcs.keys())

        # Combine all
        return list(set(backend_attrs + module_attrs + registry_attrs))

    def __getitem__(self, key: str):
        """Support subscript access for compatibility."""
        return self.__getattr__(key)

    @property
    def __all__(self):
        """Provide __all__ for 'from module import *'."""
        return self.__dir__()

    @property
    def backend(self) -> TEXBackendBase:
        """Get the underlying backend."""
        return self._backend

    def flash_attention(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> "FlashAttentionBase":
        """
        Factory method to create FlashAttention instance for the current backend.

        Usage:
            import transformer_engine_fl_torch as tex
            flash_attn = tex.flash_attention(softmax_scale=0.125)
            output = flash_attn(query, key, value, ...)

        Args:
            softmax_scale: Scale factor for softmax (typically 1/sqrt(head_dim))
            attention_dropout: Dropout probability (0.0 = no dropout)
            attention_dropout_ctx: Context manager for dropout
            attention_type: "self" or "cross" attention
            layer_number: Layer number for debugging/logging
            deterministic: Use deterministic operations

        Returns:
            FlashAttention instance for the current backend
        """
        flash_attn_class = self._backend.get_flash_attention_class()
        return flash_attn_class(
            softmax_scale=softmax_scale,
            attention_dropout=attention_dropout,
            attention_dropout_ctx=attention_dropout_ctx,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=deterministic,
        )

    def __repr__(self) -> str:
        return f"TEXModule(backend={self._backend.name})"
