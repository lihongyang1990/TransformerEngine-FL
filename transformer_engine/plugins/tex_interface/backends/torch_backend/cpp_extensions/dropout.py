# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Dropout functions implemented in pure PyTorch.

This module provides PyTorch implementations of dropout operations.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    "dropout_fwd_torch",
    "dropout_bwd_torch",
]


def dropout_fwd_torch(
    input: torch.Tensor,
    dropout_probability: float,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dropout forward pass.

    Args:
        input: Input tensor
        dropout_probability: Probability of dropping an element
        out: Pre-allocated output tensor (optional)

    Returns:
        Tuple of (output, mask)
        - output: Tensor with dropout applied
        - mask: Binary mask indicating which elements were kept
    """
    if dropout_probability == 0.0:
        # No dropout
        output = input.clone() if out is None else input.clone().to(out)
        mask = torch.ones_like(input, dtype=torch.uint8)
        return output, mask

    # Generate dropout mask
    mask = torch.bernoulli(
        torch.full_like(input, 1.0 - dropout_probability)
    ).to(torch.uint8)

    # Apply dropout with scaling
    # During training, we scale by 1/(1-p) to maintain expected value
    scale = 1.0 / (1.0 - dropout_probability)
    output = input * mask.to(input.dtype) * scale

    if out is not None:
        out.copy_(output)
        output = out

    return output, mask


def dropout_bwd_torch(
    grad_output: torch.Tensor,
    mask: torch.Tensor,
    dropout_probability: float,
    grad_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dropout backward pass.

    Args:
        grad_output: Gradient of output
        mask: Dropout mask from forward pass
        dropout_probability: Probability of dropping an element
        grad_input: Pre-allocated gradient input tensor (optional)

    Returns:
        Gradient of input
    """
    if dropout_probability == 0.0:
        # No dropout, gradient passes through unchanged
        return grad_output.clone() if grad_input is None else grad_output.clone().to(grad_input)

    # Apply mask and scale
    scale = 1.0 / (1.0 - dropout_probability)
    grad = grad_output * mask.to(grad_output.dtype) * scale

    if grad_input is not None:
        grad_input.copy_(grad)
        grad = grad_input

    return grad
