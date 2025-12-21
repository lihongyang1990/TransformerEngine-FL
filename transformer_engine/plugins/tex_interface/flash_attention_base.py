# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
FlashAttention Base Class

This module defines the abstract base class for FlashAttention implementations.
All backends should inherit from this class and implement the forward() method.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch


class FlashAttentionBase(torch.nn.Module, ABC):
    """
    Abstract base class for FlashAttention implementations.

    All backends must inherit from this class and implement the forward() method
    with the signature defined below.

    Example:
        class FlashAttentionMyBackend(FlashAttentionBase):
            def __init__(self, softmax_scale, ...):
                super().__init__(softmax_scale, ...)
                # Backend-specific initialization

            def forward(self, query_layer, key_layer, value_layer, **kwargs):
                # Backend-specific implementation
                return output
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        """
        Initialize FlashAttention base.

        Args:
            softmax_scale: Scale factor for softmax (typically 1/sqrt(head_dim))
            attention_dropout: Dropout probability (0.0 = no dropout)
            attention_dropout_ctx: Context manager for dropout (default: nullcontext)
            attention_type: "self" or "cross" attention
            layer_number: Layer number for debugging/logging
            deterministic: Use deterministic operations
        """
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.attention_dropout_ctx = attention_dropout_ctx or nullcontext
        self.attention_type = attention_type
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic

    @abstractmethod
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Any] = None,
        cp_global_ranks: Optional[List[int]] = None,
        cp_stream: Optional[torch.cuda.Stream] = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers: Optional[Any] = None,
        inference_params: Optional[Any] = None,
        flash_attention_backend: Optional[Any] = None,
        fp8_output: bool = False,
    ) -> torch.Tensor:
        """
        FlashAttention forward pass.

        Args:
            query_layer: Query tensor, shape [seq_len, batch, num_heads, head_dim] for "sbhd"
            key_layer: Key tensor, same shape as query
            value_layer: Value tensor, same shape as query
            attention_mask: Optional attention mask
            qkv_layout: Layout of QKV tensors ("sbh3d", "bsh3d", "thd", etc.)
            cu_seqlens_q: Cumulative sequence lengths for query
            cu_seqlens_kv: Cumulative sequence lengths for key/value
            max_seqlen_q: Maximum sequence length for query
            max_seqlen_kv: Maximum sequence length for key/value
            attn_mask_type: Attention mask type ("causal", "padding", etc.)
            window_size: Sliding window size for local attention
            alibi_slopes: ALiBi slopes for positional bias
            cp_group: Context parallel group
            cp_global_ranks: Context parallel global ranks
            cp_stream: Context parallel CUDA stream
            cp_comm_type: Context parallel communication type
            fp8: Use FP8 precision
            fp8_meta: FP8 metadata
            quantizers: Quantizers for FP8
            inference_params: Inference parameters (KV cache, etc.)
            flash_attention_backend: Backend version info
            fp8_output: Output in FP8

        Returns:
            Attention output tensor, shape [seq_len, batch, num_heads * head_dim]
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @property
    def backend_name(self) -> str:
        """
        Return the name of this FlashAttention backend.

        Subclasses should override this to return a descriptive name.
        """
        return self.__class__.__name__
