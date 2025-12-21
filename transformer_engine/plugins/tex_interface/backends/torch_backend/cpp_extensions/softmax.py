# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Softmax functions implemented in pure PyTorch.

This module provides PyTorch implementations of various softmax operations
including scaled softmax, masked softmax, and causal masked softmax.
"""

from typing import Optional
import torch
import torch.nn.functional as F

__all__ = [
    "scaled_softmax_forward_torch",
    "scaled_softmax_backward_torch",
    "scaled_masked_softmax_forward_torch",
    "scaled_masked_softmax_backward_torch",
    "scaled_upper_triang_masked_softmax_forward_torch",
    "scaled_upper_triang_masked_softmax_backward_torch",
    "scaled_aligned_causal_masked_softmax_forward_torch",
    "scaled_aligned_causal_masked_softmax_backward_torch",
]


def scaled_softmax_forward_torch(input: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Scaled softmax forward pass.

    Softmax(scale * input)

    Args:
        input: Input tensor
        scale: Scaling factor applied before softmax

    Returns:
        Output tensor after scaled softmax
    """
    return F.softmax(input * scale, dim=-1)


def scaled_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Scaled softmax backward pass.

    Args:
        output_grad: Gradient of output
        softmax_output: Output from forward pass
        scale: Scaling factor used in forward pass

    Returns:
        Gradient of input
    """
    # Gradient of softmax: softmax * (grad - (softmax * grad).sum(dim=-1, keepdim=True))
    grad_softmax = softmax_output * (
        output_grad - (softmax_output * output_grad).sum(dim=-1, keepdim=True)
    )

    # Apply scale
    return grad_softmax * scale


def scaled_masked_softmax_forward_torch(
    input: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Scaled masked softmax forward pass.

    Softmax(scale * input + mask)

    Args:
        input: Input tensor
        mask: Attention mask (typically 0 for valid, large negative for invalid)
        scale: Scaling factor applied before softmax

    Returns:
        Output tensor after scaled masked softmax
    """
    # Scale input and add mask
    scaled_input = input * scale + mask

    # Apply softmax
    return F.softmax(scaled_input, dim=-1)


def scaled_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Scaled masked softmax backward pass.

    Args:
        output_grad: Gradient of output
        softmax_output: Output from forward pass
        scale: Scaling factor used in forward pass

    Returns:
        Gradient of input
    """
    # Same as scaled_softmax_backward
    grad_softmax = softmax_output * (
        output_grad - (softmax_output * output_grad).sum(dim=-1, keepdim=True)
    )

    return grad_softmax * scale


def scaled_upper_triang_masked_softmax_forward_torch(
    input: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Scaled upper triangular masked softmax forward pass.

    Apply causal mask (upper triangular mask) before softmax.

    Args:
        input: Input tensor [..., seq_len, seq_len]
        scale: Scaling factor applied before softmax

    Returns:
        Output tensor after scaled causal masked softmax
    """
    seq_len = input.size(-1)

    # Create causal mask (lower triangular, including diagonal)
    # mask[i, j] = 0 if i >= j, -inf if i < j
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=input.device, dtype=input.dtype),
        diagonal=1
    )

    # Scale input and add mask
    scaled_input = input * scale + causal_mask

    # Apply softmax
    return F.softmax(scaled_input, dim=-1)


def scaled_upper_triang_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Scaled upper triangular masked softmax backward pass.

    Args:
        output_grad: Gradient of output
        softmax_output: Output from forward pass
        scale: Scaling factor used in forward pass

    Returns:
        Gradient of input
    """
    # Same gradient computation as other softmax variants
    grad_softmax = softmax_output * (
        output_grad - (softmax_output * output_grad).sum(dim=-1, keepdim=True)
    )

    return grad_softmax * scale


def scaled_aligned_causal_masked_softmax_forward_torch(
    input: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Scaled aligned causal masked softmax forward pass.

    Similar to upper triangular masked softmax but optimized for aligned sequences.

    Args:
        input: Input tensor [..., seq_len, seq_len]
        scale: Scaling factor applied before softmax

    Returns:
        Output tensor after scaled causal masked softmax
    """
    # This is essentially the same as scaled_upper_triang_masked_softmax_forward
    # The "aligned" version may have different memory layout optimizations in CUDA,
    # but functionally it's the same for PyTorch
    return scaled_upper_triang_masked_softmax_forward_torch(input, scale)


def scaled_aligned_causal_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Scaled aligned causal masked softmax backward pass.

    Args:
        output_grad: Gradient of output
        softmax_output: Output from forward pass
        scale: Scaling factor used in forward pass

    Returns:
        Gradient of input
    """
    # Same as other softmax backward passes
    grad_softmax = softmax_output * (
        output_grad - (softmax_output * output_grad).sum(dim=-1, keepdim=True)
    )

    return grad_softmax * scale
