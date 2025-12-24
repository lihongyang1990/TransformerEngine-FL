# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from transformer_engine.plugins.transformer_engine_fl.flash_attention_base import FlashAttentionBase


class FlashAttentionTorch(FlashAttentionBase):
    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            softmax_scale=softmax_scale,
            attention_dropout=attention_dropout,
            attention_dropout_ctx=attention_dropout_ctx,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=deterministic,
        )

        try:
            from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                FlashAttention as FlashAttentionNative,
            )

            self._native_flash_attn = FlashAttentionNative(
                softmax_scale=softmax_scale,
                attention_dropout=attention_dropout,
                attention_dropout_ctx=attention_dropout_ctx or nullcontext,
                attention_type=attention_type,
                layer_number=layer_number,
                deterministic=deterministic,
            )

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import native FlashAttention: {e}. "
                "Please ensure flash-attn is installed."
            )

    @property
    def backend_name(self) -> str:
        return "torch"

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
        return self._native_flash_attn(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask=attention_mask,
            qkv_layout=qkv_layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
            cp_stream=cp_stream,
            cp_comm_type=cp_comm_type,
            fp8=fp8,
            fp8_meta=fp8_meta,
            quantizers=quantizers,
            inference_params=inference_params,
            flash_attention_backend=flash_attention_backend,
            fp8_output=fp8_output,
        )
