# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformer_engine.plugin.core.ops import (
    FlashAttentionBase,
    UnfusedDotProductAttentionBase,
    FusedAttentionBase,
)


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

    @property
    def backend_name(self) -> str:
        return "torch_sdpa"

    def _convert_layout_to_bhsd(
        self,
        tensor: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Convert tensor from various layouts to [batch, heads, seq, dim] format."""
        layout = layout.lower()

        if layout in ("sbhd", "sbh3d", "sb3hd"):
            return tensor.permute(1, 2, 0, 3)
        elif layout in ("bshd", "bsh3d", "bs3hd"):
            return tensor.permute(0, 2, 1, 3)
        elif layout == "bhsd":
            return tensor
        else:
            raise ValueError(f"Unsupported qkv_layout: {layout}")

    def _convert_bhsd_to_layout(
        self,
        tensor: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Convert tensor from [batch, heads, seq, dim] back to original layout."""
        layout = layout.lower()

        if layout in ("sbhd", "sbh3d", "sb3hd"):
            return tensor.permute(2, 0, 1, 3)
        elif layout in ("bshd", "bsh3d", "bs3hd"):
            return tensor.permute(0, 2, 1, 3)
        elif layout == "bhsd":
            return tensor
        else:
            raise ValueError(f"Unsupported qkv_layout: {layout}")

    def _create_sliding_window_mask(
        self,
        seq_len_q: int,
        seq_len_kv: int,
        window_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a sliding window attention mask."""
        left_window, right_window = window_size

        if left_window == -1 and right_window == -1:
            return torch.zeros(seq_len_q, seq_len_kv, dtype=dtype, device=device)

        q_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
        kv_idx = torch.arange(seq_len_kv, device=device).unsqueeze(0)

        mask_bool = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.bool, device=device)

        if left_window >= 0:
            mask_bool = mask_bool | (kv_idx < q_idx - left_window)

        if right_window >= 0:
            mask_bool = mask_bool | (kv_idx > q_idx + right_window)

        mask = torch.zeros(seq_len_q, seq_len_kv, dtype=dtype, device=device)
        mask.masked_fill_(mask_bool, float('-inf'))

        return mask

    def _unpack_tensor(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert packed tensor to padded tensor format."""
        batch_size = cu_seqlens.shape[0] - 1
        device = tensor.device
        original_shape = tensor.shape

        if tensor.dim() == 4:
            if tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            else:
                raise ValueError(
                    f"Unexpected 4D tensor shape {original_shape}. "
                    f"Expected [total_tokens, 1, num_heads, head_dim]"
                )

        if tensor.dim() != 3:
            raise ValueError(
                f"Expected tensor to be 3D or 4D after processing, got shape {original_shape}"
            )

        total_tokens, num_heads, head_dim = tensor.shape

        expected_total = cu_seqlens[-1].item()
        if total_tokens != expected_total:
            raise ValueError(
                f"Tensor has {total_tokens} tokens but cu_seqlens indicates {expected_total} tokens"
            )

        padded_tensor = torch.zeros(
            batch_size, num_heads, max_seqlen, head_dim,
            dtype=tensor.dtype, device=device
        )

        padding_mask = torch.ones(batch_size, max_seqlen, dtype=torch.bool, device=device)

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            seq_data = tensor[start:end].permute(1, 0, 2)
            padded_tensor[i, :, :seq_len, :] = seq_data
            padding_mask[i, :seq_len] = False

        return padded_tensor, padding_mask

    def _pack_tensor(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Convert padded tensor back to packed tensor format."""
        batch_size = tensor.shape[0]
        num_heads = tensor.shape[1]
        head_dim = tensor.shape[3]
        total_tokens = cu_seqlens[-1].item()
        device = tensor.device

        packed_tensor = torch.zeros(
            total_tokens, num_heads, head_dim,
            dtype=tensor.dtype, device=device
        )

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            seq_data = tensor[i, :, :seq_len, :].permute(1, 0, 2)
            packed_tensor[start:end, :, :] = seq_data

        return packed_tensor

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
        """Flash Attention implementation using PyTorch's scaled_dot_product_attention."""
        if fp8:
            raise NotImplementedError("FP8 is not supported in PyTorch SDPA backend")
        if cp_group is not None:
            raise NotImplementedError("Context parallelism is not supported in PyTorch SDPA backend")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes are not supported in PyTorch SDPA backend")

        use_packed_format = cu_seqlens_q is not None or cu_seqlens_kv is not None
        padding_mask_q = None
        padding_mask_kv = None
        query_original_shape = query_layer.shape

        if use_packed_format:
            if cu_seqlens_q is not None:
                query, padding_mask_q = self._unpack_tensor(query_layer, cu_seqlens_q, max_seqlen_q)
            else:
                query = self._convert_layout_to_bhsd(query_layer, qkv_layout)

            if cu_seqlens_kv is not None:
                key, padding_mask_kv = self._unpack_tensor(key_layer, cu_seqlens_kv, max_seqlen_kv)
                value, _ = self._unpack_tensor(value_layer, cu_seqlens_kv, max_seqlen_kv)
            else:
                key = self._convert_layout_to_bhsd(key_layer, qkv_layout)
                value = self._convert_layout_to_bhsd(value_layer, qkv_layout)
        else:
            query = self._convert_layout_to_bhsd(query_layer, qkv_layout)
            key = self._convert_layout_to_bhsd(key_layer, qkv_layout)
            value = self._convert_layout_to_bhsd(value_layer, qkv_layout)

        batch_size, num_heads_q, seq_len_q, head_dim = query.shape
        num_heads_kv = key.shape[1]
        seq_len_kv = key.shape[2]

        if num_heads_q != num_heads_kv:
            num_groups = num_heads_q // num_heads_kv
            if num_heads_q % num_heads_kv != 0:
                raise ValueError(
                    f"num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
                )
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)

        attn_mask = None
        is_causal = False

        if use_packed_format and padding_mask_kv is not None:
            attn_mask = torch.zeros(
                batch_size, seq_len_q, seq_len_kv,
                dtype=query.dtype, device=query.device
            )
            padding_broadcast = padding_mask_kv.unsqueeze(1)
            attn_mask.masked_fill_(padding_broadcast, float('-inf'))

        if attn_mask_type == "causal":
            if window_size is None and not use_packed_format:
                is_causal = True
            else:
                causal_mask = torch.zeros(
                    seq_len_q, seq_len_kv,
                    dtype=query.dtype, device=query.device
                )
                causal_mask.masked_fill_(
                    torch.triu(torch.ones(seq_len_q, seq_len_kv, device=query.device, dtype=torch.bool), diagonal=1),
                    float('-inf')
                )

                if attn_mask is not None:
                    if attn_mask.dim() == 2:
                        attn_mask = attn_mask + causal_mask
                    else:
                        attn_mask = attn_mask + causal_mask.unsqueeze(0)
                else:
                    attn_mask = causal_mask

        if window_size is not None and not is_causal:
            window_mask = self._create_sliding_window_mask(
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                window_size=window_size,
                device=query.device,
                dtype=query.dtype,
            )

            if attn_mask is not None:
                attn_mask = attn_mask + window_mask.unsqueeze(0)
            else:
                attn_mask = window_mask

        if attention_mask is not None and attn_mask_type != "causal":
            if isinstance(attention_mask, tuple):
                explicit_mask = attention_mask[0]
            else:
                explicit_mask = attention_mask

            if explicit_mask.dtype == torch.bool:
                float_mask = torch.zeros_like(explicit_mask, dtype=query.dtype)
                float_mask.masked_fill_(~explicit_mask, float('-inf'))
                explicit_mask = float_mask

            if explicit_mask.dim() == 2:
                explicit_mask = explicit_mask.unsqueeze(0).unsqueeze(0)
            elif explicit_mask.dim() == 3:
                explicit_mask = explicit_mask.unsqueeze(1)

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                attn_mask = attn_mask + explicit_mask
            else:
                attn_mask = explicit_mask
        elif attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        with self.attention_dropout_ctx():
            dropout_p = self.attention_dropout if self.training else 0.0

            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self.softmax_scale,
            )

        if use_packed_format and padding_mask_q is not None:
            mask_expanded = padding_mask_q.unsqueeze(1).unsqueeze(3)
            output = output.masked_fill(mask_expanded, 0.0)

        if use_packed_format and cu_seqlens_q is not None:
            output = self._pack_tensor(output, cu_seqlens_q)

            if len(query_original_shape) == 4:
                total_tokens = output.shape[0]
                hidden_size = output.shape[1] * output.shape[2]
                output = output.contiguous().view(total_tokens, 1, hidden_size)
        else:
            output = self._convert_bhsd_to_layout(output, qkv_layout)
            # Flatten the last two dimensions (heads, dim) -> (heads * dim)
            # to match the output format of other backends
            output = output.contiguous().view(*output.shape[:-2], -1)

        return output


class UnfusedDotProductAttentionTorch(UnfusedDotProductAttentionBase):
    """
    Reference PyTorch implementation of UnfusedDotProductAttention.
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_type: str = "self",
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        softmax_type: str = "vanilla",
        return_max_logit: Optional[bool] = False,
    ) -> None:
        super().__init__(
            softmax_scale=softmax_scale,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            attention_dropout_ctx=attention_dropout_ctx,
            layer_number=layer_number,
            softmax_type=softmax_type,
            return_max_logit=return_max_logit,
        )
        self.dropout = torch.nn.Dropout(attention_dropout)

    @property
    def backend_name(self) -> str:
        return "torch_unfused"

    def _convert_to_sbhd(
        self,
        tensor: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Convert tensor to [seq, batch, heads, dim] format."""
        layout = layout.lower()

        if layout in ("sbhd", "sbh3d", "sb3hd"):
            return tensor
        elif layout in ("bshd", "bsh3d", "bs3hd"):
            return tensor.transpose(0, 1)
        else:
            raise ValueError(f"Unsupported qkv_layout: {layout}")

    def _convert_from_sbhd(
        self,
        tensor: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Convert tensor from [seq, batch, heads, dim] back to original layout."""
        layout = layout.lower()

        if layout in ("sbhd", "sbh3d", "sb3hd"):
            return tensor
        elif layout in ("bshd", "bsh3d", "bs3hd"):
            return tensor.transpose(0, 1)
        else:
            raise ValueError(f"Unsupported qkv_layout: {layout}")

    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        seq_len_q: int,
        seq_len_kv: int,
    ) -> torch.Tensor:
        """Apply causal mask to attention scores."""
        mask = torch.triu(
            torch.ones(seq_len_q, seq_len_kv, device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))
        return scores

    def forward(
        self,
        _alibi_cache: Dict[str, Any],
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[torch.Tensor] = None,
        max_seqlen_kv: Optional[torch.Tensor] = None,
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        window_size: Optional[Tuple[int, int]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        inference_params: Optional[Any] = None,
        softmax_offset: torch.Tensor = None,
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers: Optional[Any] = None,
        fp8_output: bool = False,
    ) -> torch.Tensor:
        """Unfused attention forward pass using pure PyTorch."""
        if fp8:
            raise NotImplementedError("FP8 is not supported in PyTorch reference backend")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes are not supported in this implementation")
        if core_attention_bias_type != "no_bias":
            raise NotImplementedError("Attention bias is not supported in this implementation")
        if window_size is not None:
            raise NotImplementedError("Sliding window attention is not supported in this implementation")

        # Convert to sbhd format
        query = self._convert_to_sbhd(query_layer, qkv_layout)
        key = self._convert_to_sbhd(key_layer, qkv_layout)
        value = self._convert_to_sbhd(value_layer, qkv_layout)

        # Get dimensions: [seq, batch, heads, dim]
        seq_len_q, batch_size, num_heads_q, head_dim = query.shape
        seq_len_kv = key.shape[0]
        num_heads_kv = key.shape[2]

        # Handle GQA (Grouped Query Attention)
        if num_heads_q != num_heads_kv:
            assert num_heads_q % num_heads_kv == 0, \
                "num_heads_q must be divisible by num_heads_kv for GQA"
            num_groups = num_heads_q // num_heads_kv
            key = key.repeat_interleave(num_groups, dim=2)
            value = value.repeat_interleave(num_groups, dim=2)

        # Reshape for batched matrix multiplication
        # [seq, batch, heads, dim] -> [batch * heads, seq, dim]
        query = query.permute(1, 2, 0, 3).reshape(batch_size * num_heads_q, seq_len_q, head_dim)
        key = key.permute(1, 2, 0, 3).reshape(batch_size * num_heads_q, seq_len_kv, head_dim)
        value = value.permute(1, 2, 0, 3).reshape(batch_size * num_heads_q, seq_len_kv, head_dim)

        # Compute attention scores: Q @ K^T
        # [batch * heads, seq_q, dim] @ [batch * heads, dim, seq_kv] -> [batch * heads, seq_q, seq_kv]
        scores = torch.bmm(query, key.transpose(1, 2)) * self.softmax_scale

        # Reshape back to apply masks
        # [batch * heads, seq_q, seq_kv] -> [batch, heads, seq_q, seq_kv]
        scores = scores.view(batch_size, num_heads_q, seq_len_q, seq_len_kv)

        # Apply causal mask
        if "causal" in attn_mask_type:
            scores = self._apply_causal_mask(scores, seq_len_q, seq_len_kv)

        # Apply custom attention mask
        if attention_mask is not None:
            if isinstance(attention_mask, tuple):
                mask = attention_mask[0]
            else:
                mask = attention_mask

            # Handle boolean masks
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float('-inf'))
            else:
                scores = scores + mask

        # Softmax
        # [batch, heads, seq_q, seq_kv]
        attention_probs = F.softmax(scores, dim=-1)

        # Apply dropout
        with self.attention_dropout_ctx():
            attention_probs = self.dropout(attention_probs)

        # Reshape for matmul
        # [batch, heads, seq_q, seq_kv] -> [batch * heads, seq_q, seq_kv]
        attention_probs = attention_probs.view(batch_size * num_heads_q, seq_len_q, seq_len_kv)

        # Compute output: attention_probs @ V
        # [batch * heads, seq_q, seq_kv] @ [batch * heads, seq_kv, dim] -> [batch * heads, seq_q, dim]
        output = torch.bmm(attention_probs, value)

        # Reshape and convert back to original layout
        # [batch * heads, seq_q, dim] -> [seq_q, batch, heads, dim]
        output = output.view(batch_size, num_heads_q, seq_len_q, head_dim)
        output = output.permute(2, 0, 1, 3).contiguous()

        # Convert back to original layout
        output = self._convert_from_sbhd(output, qkv_layout)

        # Flatten last two dimensions
        # [seq, batch, heads, dim] -> [seq, batch, heads * dim]
        output = output.view(*output.shape[:-2], -1)

        return output


class FusedAttentionTorch(FusedAttentionBase):
    """
    Reference PyTorch implementation of FusedAttention using PyTorch SDPA.
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
        softmax_type: str = "vanilla",
        return_max_logit: Optional[bool] = False,
    ) -> None:
        super().__init__(
            softmax_scale=softmax_scale,
            attention_dropout=attention_dropout,
            attention_dropout_ctx=attention_dropout_ctx,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=deterministic,
            softmax_type=softmax_type,
            return_max_logit=return_max_logit,
        )

    @property
    def backend_name(self) -> str:
        return "torch_fused_sdpa"

    def _convert_to_bhsd(
        self,
        tensor: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Convert tensor to [batch, heads, seq, dim] format."""
        layout = layout.lower()

        if layout in ("sbhd", "sbh3d", "sb3hd"):
            # [seq, batch, heads, dim] -> [batch, heads, seq, dim]
            return tensor.permute(1, 2, 0, 3)
        elif layout in ("bshd", "bsh3d", "bs3hd"):
            # [batch, seq, heads, dim] -> [batch, heads, seq, dim]
            return tensor.permute(0, 2, 1, 3)
        elif layout == "bhsd":
            return tensor
        else:
            raise ValueError(f"Unsupported qkv_layout: {layout}")

    def _convert_from_bhsd(
        self,
        tensor: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Convert tensor from [batch, heads, seq, dim] back to original layout."""
        layout = layout.lower()

        if layout in ("sbhd", "sbh3d", "sb3hd"):
            # [batch, heads, seq, dim] -> [seq, batch, heads, dim]
            return tensor.permute(2, 0, 1, 3)
        elif layout in ("bshd", "bsh3d", "bs3hd"):
            # [batch, heads, seq, dim] -> [batch, seq, heads, dim]
            return tensor.permute(0, 2, 1, 3)
        elif layout == "bhsd":
            return tensor
        else:
            raise ValueError(f"Unsupported qkv_layout: {layout}")

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        window_size: Optional[Tuple[int, int]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Any] = None,
        cp_global_ranks: Optional[List[int]] = None,
        cp_stream: Optional[torch.cuda.Stream] = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers: Optional[Any] = None,
        fused_attention_backend: Optional[Any] = None,
        inference_params: Optional[Any] = None,
        softmax_offset: torch.Tensor = None,
        fp8_output: bool = False,
    ) -> torch.Tensor:
        """Fused attention forward pass using PyTorch SDPA."""
        if fp8:
            raise NotImplementedError("FP8 is not supported in PyTorch SDPA backend")
        if cp_group is not None:
            raise NotImplementedError("Context parallelism is not supported")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes are not supported")
        if core_attention_bias_type != "no_bias":
            raise NotImplementedError("Attention bias is not supported")
        if window_size is not None:
            raise NotImplementedError("Sliding window attention is not supported")

        # Convert to bhsd format for PyTorch SDPA
        query = self._convert_to_bhsd(query_layer, qkv_layout)
        key = self._convert_to_bhsd(key_layer, qkv_layout)
        value = self._convert_to_bhsd(value_layer, qkv_layout)

        # Get dimensions
        batch_size, num_heads_q, seq_len_q, head_dim = query.shape
        num_heads_kv = key.shape[1]

        # Handle GQA
        if num_heads_q != num_heads_kv:
            assert num_heads_q % num_heads_kv == 0
            num_groups = num_heads_q // num_heads_kv
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)

        # Prepare attention mask
        attn_mask = None
        is_causal = False

        if "causal" in attn_mask_type and attention_mask is None:
            is_causal = True
        elif attention_mask is not None:
            if isinstance(attention_mask, tuple):
                attn_mask = attention_mask[0]
            else:
                attn_mask = attention_mask

            # Convert boolean mask to additive mask
            if attn_mask.dtype == torch.bool:
                attn_mask_float = torch.zeros_like(attn_mask, dtype=query.dtype)
                attn_mask_float.masked_fill_(attn_mask, float('-inf'))
                attn_mask = attn_mask_float

        # Use PyTorch's scaled_dot_product_attention
        with self.attention_dropout_ctx():
            dropout_p = self.attention_dropout if self.training else 0.0

            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self.softmax_scale,
            )

        # Convert back to original layout
        output = self._convert_from_bhsd(output, qkv_layout)

        # Flatten last two dimensions
        output = output.contiguous().view(*output.shape[:-2], -1)

        return output
