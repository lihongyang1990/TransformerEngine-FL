# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Optimizer operations implemented in pure PyTorch.

This module provides PyTorch implementations of multi-tensor optimizer operations
including Adam, SGD, and utility functions.
"""

from typing import List, Union
import torch

__all__ = [
    "multi_tensor_scale_torch",
    "multi_tensor_l2norm_torch",
    "multi_tensor_adam_torch",
    "multi_tensor_sgd_torch",
]


def multi_tensor_scale_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    scale: float,
) -> None:
    """
    Scale multiple tensors by a scalar value.

    Args:
        chunk_size: Chunk size for processing (not used in PyTorch implementation)
        noop_flag: If non-zero, skip the operation
        tensor_lists: List containing [input_tensors, output_tensors]
                      NOTE: input first, output second (TransformerEngine convention)
        scale: Scaling factor
    """
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 2:
        raise ValueError("tensor_lists should contain [input_tensors, output_tensors]")

    # IMPORTANT: TransformerEngine uses [input, output] order
    input_tensors, output_tensors = tensor_lists

    if len(output_tensors) != len(input_tensors):
        raise ValueError("Output and input tensor lists must have the same length")

    for in_tensor, out_tensor in zip(input_tensors, output_tensors):
        out_tensor.copy_(in_tensor * scale)


def multi_tensor_l2norm_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    per_tensor: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Compute L2 norm of multiple tensors.

    Args:
        chunk_size: Chunk size for processing (not used in PyTorch implementation)
        noop_flag: If non-zero, return zero norm
        tensor_lists: List containing [tensors] to compute norm
        per_tensor: If True, return per-tensor norms; if False, return global norm

    Returns:
        L2 norm (global or per-tensor)
    """
    if noop_flag.item() != 0:
        if per_tensor:
            return [torch.tensor(0.0, device=t.device) for t in tensor_lists[0]]
        else:
            return torch.tensor(0.0, device=tensor_lists[0][0].device)

    tensors = tensor_lists[0]

    if per_tensor:
        # Return per-tensor norms
        norms = []
        for tensor in tensors:
            norm = torch.norm(tensor.float(), p=2)
            norms.append(norm)
        return norms
    else:
        # Return global norm (sqrt of sum of squares)
        total_norm_sq = torch.tensor(0.0, device=tensors[0].device)
        for tensor in tensors:
            total_norm_sq += torch.sum(tensor.float() ** 2)
        return torch.sqrt(total_norm_sq)


def multi_tensor_adam_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None:
    """
    Multi-tensor Adam optimizer step.

    Args:
        chunk_size: Chunk size for processing (not used in PyTorch implementation)
        noop_flag: If non-zero, skip the operation
        tensor_lists: List containing [grads, params, exp_avgs, exp_avg_sqs]
                      NOTE: Order is grads first, then params (TransformerEngine convention)
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        eps: Epsilon for numerical stability
        step: Current step number (for bias correction)
        mode: Adam mode (0=Adam, 1=AdamW)
        bias_correction: Whether to apply bias correction (1=yes, 0=no)
        weight_decay: Weight decay coefficient
    """
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 4:
        raise ValueError("tensor_lists should contain [grads, params, exp_avgs, exp_avg_sqs]")

    # IMPORTANT: TransformerEngine uses [grads, params, ...] order, not [params, grads, ...]
    grads, params, exp_avgs, exp_avg_sqs = tensor_lists

    if not (len(params) == len(grads) == len(exp_avgs) == len(exp_avg_sqs)):
        raise ValueError("All tensor lists must have the same length")

    # Compute bias correction terms
    if bias_correction:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
    else:
        bias_correction1 = 1.0
        bias_correction2 = 1.0

    for grad, param, exp_avg, exp_avg_sq in zip(grads, params, exp_avgs, exp_avg_sqs):
        # Skip if gradient is None
        if grad is None:
            continue

        # Apply weight decay (AdamW mode)
        if mode == 1 and weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        # Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Update biased second raw moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute bias-corrected moments
        corrected_exp_avg = exp_avg / bias_correction1
        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

        # Update parameters
        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(corrected_exp_avg, denom, value=-lr)


def multi_tensor_sgd_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
) -> None:
    """
    Multi-tensor SGD optimizer step.

    Args:
        chunk_size: Chunk size for processing (not used in PyTorch implementation)
        noop_flag: If non-zero, skip the operation
        tensor_lists: List containing [params, grads, momentum_buffers]
        lr: Learning rate
        momentum: Momentum factor
        dampening: Dampening for momentum
        weight_decay: Weight decay coefficient
        nesterov: Whether to use Nesterov momentum
    """
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 3:
        raise ValueError("tensor_lists should contain [params, grads, momentum_buffers]")

    params, grads, momentum_buffers = tensor_lists

    if not (len(params) == len(grads) == len(momentum_buffers)):
        raise ValueError("All tensor lists must have the same length")

    for param, grad, buf in zip(params, grads, momentum_buffers):
        # Skip if gradient is None
        if grad is None:
            continue

        # Apply weight decay
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Apply momentum
        if momentum != 0:
            if buf is None or buf.numel() == 0:
                # Initialize momentum buffer
                buf = grad.clone().detach()
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        # Update parameters
        param.add_(grad, alpha=-lr)
