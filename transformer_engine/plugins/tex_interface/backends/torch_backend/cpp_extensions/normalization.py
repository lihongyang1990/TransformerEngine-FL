# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Normalization functions implemented in pure PyTorch.

This module provides PyTorch implementations of LayerNorm and RMSNorm.
"""

from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    "layernorm_fwd_torch",
    "layernorm_bwd_torch",
    # rmsnorm functions are already implemented in rmsnorm.py
]


def layernorm_fwd_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    ln_out: Optional[torch.Tensor],
    quantizer: Any,
    odtype: torch.dtype,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LayerNorm forward pass implemented in PyTorch.

    Args:
        input: Input tensor
        weight: Gamma/weight parameter
        bias: Beta/bias parameter (optional)
        eps: Epsilon for numerical stability
        ln_out: Pre-allocated output tensor (optional)
        quantizer: Quantization parameters (not used in torch backend)
        odtype: Output data type
        sm_margin: SM margin for kernel launch (not used in torch backend)
        zero_centered_gamma: Whether gamma is zero-centered

    Returns:
        Tuple of (output, mean, rsigma)
    """
    # Compute mean and variance
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    rsigma = torch.rsqrt(var + eps)

    # Normalize
    normalized = (input - mean) * rsigma

    # Apply weight (gamma)
    if zero_centered_gamma:
        # gamma is zero-centered: output = normalized * (1 + gamma)
        output = normalized * (1.0 + weight)
    else:
        # standard: output = normalized * gamma
        output = normalized * weight

    # Apply bias (beta) if provided
    if bias is not None:
        output = output + bias

    # Convert to output dtype
    if output.dtype != odtype:
        output = output.to(odtype)

    # Remove keepdim for mean and rsigma outputs
    mean = mean.squeeze(-1)
    rsigma = rsigma.squeeze(-1)

    return output, mean, rsigma


def layernorm_bwd_torch(
    dy: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LayerNorm backward pass implemented in PyTorch.

    Args:
        dy: Gradient of output
        x: Input tensor from forward
        mu: Mean from forward
        rsigma: Inverse standard deviation from forward
        gamma: Weight tensor
        sm_margin: SM margin for kernel launch (not used in torch backend)
        zero_centered_gamma: Whether gamma is zero-centered

    Returns:
        Tuple of (dx, dgamma, dbeta)
    """
    # Ensure mu and rsigma have the right shape
    if mu.ndim < x.ndim:
        mu = mu.unsqueeze(-1)
    if rsigma.ndim < x.ndim:
        rsigma = rsigma.unsqueeze(-1)

    # Normalize input
    x_normalized = (x - mu) * rsigma

    # Number of elements in the normalized dimension
    N = x.shape[-1]

    # Apply gamma adjustment if zero-centered
    if zero_centered_gamma:
        gamma_adj = 1.0 + gamma
    else:
        gamma_adj = gamma

    # Gradient w.r.t. normalized input
    dy_gamma = dy * gamma_adj

    # Compute gradients
    # Mean of dy_gamma
    mean_dy_gamma = dy_gamma.mean(dim=-1, keepdim=True)

    # Mean of dy_gamma * x_normalized
    mean_dy_gamma_x = (dy_gamma * x_normalized).mean(dim=-1, keepdim=True)

    # Gradient w.r.t. input
    dx = rsigma * (dy_gamma - mean_dy_gamma - x_normalized * mean_dy_gamma_x)

    # Gradient w.r.t. gamma (sum over all dimensions except last)
    dgamma = (dy * x_normalized).sum(dim=tuple(range(dy.ndim - 1)))

    # Gradient w.r.t. beta (sum over all dimensions except last)
    dbeta = dy.sum(dim=tuple(range(dy.ndim - 1)))

    return dx, dgamma, dbeta
