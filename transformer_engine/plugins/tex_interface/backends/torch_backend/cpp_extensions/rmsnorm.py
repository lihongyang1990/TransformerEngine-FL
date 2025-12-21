# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch

__all__ = [
    "rmsnorm_fwd_torch",
    "rmsnorm_bwd_torch",
]


def rmsnorm_fwd_torch(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    odtype,
    sm_margin,
    zero_centered_gamma,
):
    """RMSNorm forward using PyTorch native operations."""
    # Ensure weight is on the same device as input
    if weight.device != input.device:
        weight = weight.to(input.device)

    # Compute variance (mean of squares)
    variance = input.pow(2).mean(-1, keepdim=True)
    # Compute inverse RMS
    inv_rms = torch.rsqrt(variance + eps)
    # Normalize
    y = input * inv_rms
    # Apply weight
    if zero_centered_gamma:
        y = y * (1 + weight)
    else:
        y = y * weight

    # Return: output, None (for quantized output), inverse RMS (for backward)
    # inv_rms shape needs to be squeezed for backward compatibility
    rstdevs = inv_rms.squeeze(-1)

    return y, None, rstdevs


def rmsnorm_bwd_torch(
    dy,
    x,
    rsigma,
    gamma,
    sm_margin,
    zero_centered_gamma,
    eps,
):
    """RMSNorm backward using PyTorch native operations."""
    # rsigma is the inverse RMS from forward pass, shape: [...] (without last dim)
    # Need to unsqueeze for broadcasting
    inv_rms = rsigma.unsqueeze(-1)

    # Compute normalized x
    x_norm = x * inv_rms

    # Apply zero_centered_gamma if needed
    if zero_centered_gamma:
        weight = 1 + gamma
    else:
        weight = gamma

    # Gradient w.r.t. weight: sum over all dims except last
    # dw = sum(dy * x_norm)
    dw = (dy * x_norm).sum(dim=tuple(range(dy.ndim - 1)))

    # Gradient w.r.t. input
    # d(x * inv_rms * w) / dx = inv_rms * w + x * d(inv_rms)/dx * w
    # where inv_rms = 1/sqrt(mean(x^2) + eps)
    # d(inv_rms)/dx = -0.5 * inv_rms^3 * 2x/N = -inv_rms^3 * x/N

    dy_weighted = dy * weight

    # Simple gradient computation
    # dx = dy * w * inv_rms - x * inv_rms^3 * mean(dy * w * x)
    mean_term = (dy_weighted * x_norm).mean(-1, keepdim=True)
    dx = inv_rms * (dy_weighted - x_norm * mean_term)
    return dx, dw
