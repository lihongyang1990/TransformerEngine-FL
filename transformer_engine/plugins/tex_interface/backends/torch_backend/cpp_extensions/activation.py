# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Activation functions implemented in pure PyTorch.

This module provides PyTorch implementations of various activation functions
including GELU, SiLU/Swish, ReLU, and their gated variants (GLU).
"""

from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    # Forward activations
    "gelu_torch",
    "geglu_torch",
    "qgelu_torch",
    "qgeglu_torch",
    "relu_torch",
    "reglu_torch",
    "srelu_torch",
    "sreglu_torch",
    "silu_torch",
    "swiglu_torch",
    "clamped_swiglu_torch",
    # Backward activations
    "dgelu_torch",
    "dgeglu_torch",
    "dqgelu_torch",
    "dqgeglu_torch",
    "drelu_torch",
    "dreglu_torch",
    "dsrelu_torch",
    "dsreglu_torch",
    "dsilu_torch",
    "dswiglu_torch",
    "clamped_dswiglu_torch",
    # DBias + DAct fusions
    "dbias_dgelu_torch",
    "dbias_dsilu_torch",
    "dbias_drelu_torch",
    "dbias_dqgelu_torch",
    "dbias_dsrelu_torch",
]


# ==============================================================================
# Forward Activation Functions
# ==============================================================================

def gelu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    GELU activation (tanh approximation).

    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    """
    # Tanh approximation of GELU
    return F.gelu(input, approximate='tanh')


def geglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gated GELU activation.

    Input is split into two halves along the last dimension.
    Output = GELU(first_half) * second_half
    """
    a, b = input.chunk(2, dim=-1)
    return F.gelu(a, approximate='tanh') * b


def qgelu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Quick GELU activation.

    QGELU(x) = x * sigmoid(1.702 * x)
    """
    return input * torch.sigmoid(1.702 * input)


def qgeglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gated Quick GELU activation.

    Input is split into two halves along the last dimension.
    Output = QGELU(first_half) * second_half
    """
    a, b = input.chunk(2, dim=-1)
    return a * torch.sigmoid(1.702 * a) * b


def relu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    ReLU activation.

    ReLU(x) = max(x, 0)
    """
    return F.relu(input)


def reglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gated ReLU activation.

    Input is split into two halves along the last dimension.
    Output = ReLU(first_half) * second_half
    """
    a, b = input.chunk(2, dim=-1)
    return F.relu(a) * b


def srelu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Squared ReLU activation.

    SReLU(x) = ReLU(x)^2
    """
    return torch.square(F.relu(input))


def sreglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gated Squared ReLU activation.

    Input is split into two halves along the last dimension.
    Output = SReLU(first_half) * second_half
    """
    a, b = input.chunk(2, dim=-1)
    return torch.square(F.relu(a)) * b


def silu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    SiLU (Swish) activation.

    SiLU(x) = x * sigmoid(x)
    """
    return F.silu(input)


def swiglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gated SiLU (Swish) activation.

    Input is split into two halves along the last dimension.
    Output = SiLU(first_half) * second_half
    """
    a, b = input.chunk(2, dim=-1)
    return F.silu(a) * b


def clamped_swiglu_torch(
    input: torch.Tensor,
    quantizer: Any,
    limit: float = 7.0,
    alpha: float = 1.702,
) -> torch.Tensor:
    """
    Clamped SwiGLU activation.

    Both gate and pre-activations are clamped to [-limit, limit].
    Uses sigmoid(alpha * x) instead of sigmoid(x).
    """
    a, b = input.chunk(2, dim=-1)
    # Clamp both parts
    a = torch.clamp(a, -limit, limit)
    b = torch.clamp(b, -limit, limit)
    # Apply gated activation with alpha scaling
    return a * torch.sigmoid(alpha * a) * b


# ==============================================================================
# Backward Activation Functions
# ==============================================================================

def dgelu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of GELU activation.
    """
    # Enable gradient computation
    x = fwd_input.detach().requires_grad_(True)
    with torch.enable_grad():
        y = F.gelu(x, approximate='tanh')
        y.backward(grad)
    return x.grad


def dgeglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of GEGLU activation.
    """
    a, b = fwd_input.chunk(2, dim=-1)
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    with torch.enable_grad():
        y = F.gelu(a, approximate='tanh') * b
        y.backward(grad)

    return torch.cat([a.grad, b.grad], dim=-1)


def dqgelu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of Quick GELU activation.
    """
    x = fwd_input.detach().requires_grad_(True)
    with torch.enable_grad():
        y = x * torch.sigmoid(1.702 * x)
        y.backward(grad)
    return x.grad


def dqgeglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of Quick GEGLU activation.
    """
    a, b = fwd_input.chunk(2, dim=-1)
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    with torch.enable_grad():
        y = a * torch.sigmoid(1.702 * a) * b
        y.backward(grad)

    return torch.cat([a.grad, b.grad], dim=-1)


def drelu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of ReLU activation.
    """
    return grad * (fwd_input > 0).to(grad.dtype)


def dreglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of ReGLU activation.
    """
    a, b = fwd_input.chunk(2, dim=-1)

    # Gradient w.r.t. a: grad * b * (a > 0)
    grad_a = grad * b * (a > 0).to(grad.dtype)
    # Gradient w.r.t. b: grad * ReLU(a)
    grad_b = grad * F.relu(a)

    return torch.cat([grad_a, grad_b], dim=-1)


def dsrelu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of Squared ReLU activation.

    d/dx[ReLU(x)^2] = 2 * ReLU(x) * (x > 0)
    """
    relu_x = F.relu(fwd_input)
    return 2 * grad * relu_x * (fwd_input > 0).to(grad.dtype)


def dsreglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of Squared ReGLU activation.
    """
    a, b = fwd_input.chunk(2, dim=-1)

    relu_a = F.relu(a)
    # Gradient w.r.t. a: grad * b * 2 * ReLU(a) * (a > 0)
    grad_a = grad * b * 2 * relu_a * (a > 0).to(grad.dtype)
    # Gradient w.r.t. b: grad * ReLU(a)^2
    grad_b = grad * torch.square(relu_a)

    return torch.cat([grad_a, grad_b], dim=-1)


def dsilu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of SiLU activation.
    """
    x = fwd_input.detach().requires_grad_(True)
    with torch.enable_grad():
        y = F.silu(x)
        y.backward(grad)
    return x.grad


def dswiglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    """
    Gradient of SwiGLU activation.
    """
    a, b = fwd_input.chunk(2, dim=-1)
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    with torch.enable_grad():
        y = F.silu(a) * b
        y.backward(grad)

    return torch.cat([a.grad, b.grad], dim=-1)


def clamped_dswiglu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
    limit: float = 7.0,
    alpha: float = 1.702,
) -> torch.Tensor:
    """
    Gradient of Clamped SwiGLU activation.
    """
    a, b = fwd_input.chunk(2, dim=-1)

    # Clamp both parts
    a_clamped = torch.clamp(a, -limit, limit)
    b_clamped = torch.clamp(b, -limit, limit)

    a_clamped = a_clamped.detach().requires_grad_(True)
    b_clamped = b_clamped.detach().requires_grad_(True)

    with torch.enable_grad():
        y = a_clamped * torch.sigmoid(alpha * a_clamped) * b_clamped
        y.backward(grad)

    # Zero out gradients where clamping occurred
    grad_a = a_clamped.grad * ((a >= -limit) & (a <= limit)).to(grad.dtype)
    grad_b = b_clamped.grad * ((b >= -limit) & (b <= limit)).to(grad.dtype)

    return torch.cat([grad_a, grad_b], dim=-1)


# ==============================================================================
# DBias + DAct Fusions
# ==============================================================================

def dbias_dgelu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient of bias and GELU jointly.

    Returns:
        Tuple of (grad_input, grad_bias)
    """
    # Gradient of GELU
    grad_input = dgelu_torch(grad, fwd_input, quantizer)

    # Gradient of bias (sum over all dimensions except the last)
    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias


def dbias_dsilu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient of bias and SiLU jointly.

    Returns:
        Tuple of (grad_input, grad_bias)
    """
    # Gradient of SiLU
    grad_input = dsilu_torch(grad, fwd_input, quantizer)

    # Gradient of bias
    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias


def dbias_drelu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient of bias and ReLU jointly.

    Returns:
        Tuple of (grad_input, grad_bias)
    """
    # Gradient of ReLU
    grad_input = drelu_torch(grad, fwd_input, quantizer)

    # Gradient of bias
    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias


def dbias_dqgelu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient of bias and Quick GELU jointly.

    Returns:
        Tuple of (grad_input, grad_bias)
    """
    # Gradient of Quick GELU
    grad_input = dqgelu_torch(grad, fwd_input, quantizer)

    # Gradient of bias
    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias


def dbias_dsrelu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient of bias and Squared ReLU jointly.

    Returns:
        Tuple of (grad_input, grad_bias)
    """
    # Gradient of Squared ReLU
    grad_input = dsrelu_torch(grad, fwd_input, quantizer)

    # Gradient of bias
    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias
