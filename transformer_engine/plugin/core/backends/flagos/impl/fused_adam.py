# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional, List
import torch
import flag_gems


def multi_tensor_adam_fl(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
    inv_scale: Optional[float] = 1.0,
    out_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Adam optimizer implementation matching CUDA exactly.

    mode == 0: L2 regularization (add weight_decay * param to gradient before moment update)
    mode == 1: AdamW (add weight_decay * param to update after moment computation)
    """
    num_lists = len(tensor_lists)
    assert num_lists in [4, 5], f"Expected 4 or 5 tensor lists, got {num_lists}"

    num_tensors = len(tensor_lists[0])
    assert num_tensors > 0, "No tensors provided"

    for i, lst in enumerate(tensor_lists):
        assert len(lst) == num_tensors, f"List {i} has {len(lst)} tensors, expected {num_tensors}"

    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if bias_correction == 1:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

    is_adamw = (mode == 1)

    for i in range(num_tensors):
        g = tensor_lists[0][i]
        p = tensor_lists[1][i]
        m = tensor_lists[2][i]
        v = tensor_lists[3][i]
        p_master = tensor_lists[4][i] if num_lists == 5 else None

        if not g.is_contiguous():
            g = g.contiguous()

        # Convert to float for computation (matches CUDA's MATH_T = float)
        g = g.float()
        p_float = p.float()

        if inv_scale is not None and inv_scale != 1.0:
            g = flag_gems.mul(g, inv_scale)

        if mode == 0:  # L2 regularization
            # Add weight decay to gradient before moment update
            g = flag_gems.add(g, p_float, alpha=weight_decay)

            # Update moments with modified gradient
            flag_gems.add_(flag_gems.mul_(m, beta1), g, alpha=1-beta1)
            flag_gems.add_(flag_gems.mul_(v, beta2), flag_gems.mul(g, g), alpha=1-beta2)

            # Bias correction
            m_corr = flag_gems.true_divide(m, bias_correction1)
            v_corr = flag_gems.true_divide(v, bias_correction2)

            # Compute update
            denom = flag_gems.add(flag_gems.sqrt(v_corr), epsilon)
            update = flag_gems.true_divide(m_corr, denom)

            # Update parameter
            p.add_(update, alpha=-lr)
        else:  # mode == 1, AdamW (decoupled weight decay)
            # Update moments with original gradient
            flag_gems.add_(flag_gems.mul_(m, beta1), g, alpha=1-beta1)
            flag_gems.add_(flag_gems.mul_(v, beta2), flag_gems.mul(g, g), alpha=1-beta2)

            # Bias correction
            m_corr = flag_gems.true_divide(m, bias_correction1)
            v_corr = flag_gems.true_divide(v, bias_correction2)

            # Compute update with weight decay added (matches CUDA exactly)
            denom = flag_gems.add(flag_gems.sqrt(v_corr), epsilon)
            update = flag_gems.add(flag_gems.true_divide(m_corr, denom), p_float, alpha=weight_decay)

            # Update parameter
            p.add_(update, alpha=-lr)

        if p_master is not None:
            flag_gems.copy_(p_master, p)
            out_dtype = p_master.dtype if out_dtype is None else out_dtype
            p.data = p.data.to(out_dtype)

def multi_tensor_adam_param_remainder_fl(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None:
    """
    Adam optimizer with parameter remainders for BF16 precision (FlagOS implementation).
    """
    if noop_flag.item() != 0:
        return

    num_lists = len(tensor_lists)
    assert num_lists == 5, f"Expected 5 tensor lists, got {num_lists}"

    num_tensors = len(tensor_lists[0])
    assert num_tensors > 0, "No tensors provided"

    for i, lst in enumerate(tensor_lists):
        assert len(lst) == num_tensors, f"List {i} has {len(lst)} tensors, expected {num_tensors}"

    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if bias_correction == 1:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

    is_adamw = (mode == 1)

    for i in range(num_tensors):
        g = tensor_lists[0][i]
        p = tensor_lists[1][i]  # int16 parameter (high 16 bits of FP32)
        m = tensor_lists[2][i]  # FP32 first moment
        v = tensor_lists[3][i]  # FP32 second moment
        p_remainder = tensor_lists[4][i]  # int16 remainder (low 16 bits of FP32)

        if not g.is_contiguous():
            g = g.contiguous()

        # Convert gradient to float
        g_float = g.float()

        # Reconstruct FP32 master weight from int16 param + int16 remainder using bit manipulation
        # This matches the CUDA implementation exactly:
        # 1. If p_remainder < 0, decrement p (undo rounding)
        # 2. Combine high 16 bits (p) and low 16 bits (p_remainder) into FP32
        # Note: Use PyTorch native ops for bit manipulation (int16/int32 operations)

        local_p = p.view(torch.int16).clone()
        local_p_rem = p_remainder.clone()

        # Undo rounding: if remainder < 0, decrement p
        local_p = torch.where(local_p_rem < 0, local_p - 1, local_p)

        # Combine into FP32 using bit shift operations
        # local_p is high 16 bits, local_p_rem is low 16 bits
        high_bits = local_p.to(torch.int32) << 16
        low_bits = local_p_rem.to(torch.int32) & 0xFFFF  # Mask off sign extension
        param_int32 = high_bits | low_bits
        param_master = param_int32.view(torch.float32)

        # L2 mode: add weight decay to gradient before updating moments
        if not is_adamw and weight_decay != 0:
            g_float = flag_gems.add(g_float, param_master, alpha=weight_decay)

        # Update first moment: m = beta1 * m + (1 - beta1) * g
        flag_gems.add_(flag_gems.mul_(m, beta1), g_float, alpha=1 - beta1)

        # Update second moment: v = beta2 * v + (1 - beta2) * g^2
        flag_gems.add_(flag_gems.mul_(v, beta2), flag_gems.mul(g_float, g_float), alpha=1 - beta2)

        # Apply bias correction
        m_corr = flag_gems.true_divide(m, bias_correction1)
        v_corr = flag_gems.true_divide(v, bias_correction2)

        # Compute denominator: sqrt(v_corr) + epsilon
        denom = flag_gems.add(flag_gems.sqrt(v_corr), epsilon)

        # Compute update
        update = flag_gems.true_divide(m_corr, denom)

        # AdamW mode: add decoupled weight decay to update
        if is_adamw and weight_decay != 0:
            update = flag_gems.add(update, param_master, alpha=weight_decay)

        # Update master weight: p = p - lr * update
        param_master = flag_gems.sub(param_master, flag_gems.mul(update, lr))

        # Split FP32 back into int16 param + int16 remainder using bit manipulation
        # This matches the CUDA implementation exactly:
        # 1. Extract high 16 bits as p
        # 2. Extract low 16 bits as p_remainder
        # 3. If p_remainder < 0, increment p (round up)
        # Note: Use PyTorch native ops for bit manipulation (int32 operations)

        param_int32 = param_master.view(torch.int32)
        # Extract low 16 bits (remainder) and high 16 bits (param)
        new_p_rem = (param_int32 & 0xFFFF).to(torch.int16)
        new_p = ((param_int32 >> 16) & 0xFFFF).to(torch.int16)

        # Round up: if remainder < 0, increment p
        new_p = torch.where(new_p_rem < 0, new_p + 1, new_p)

        # Write back
        p.view(torch.int16).copy_(new_p)
        p_remainder.copy_(new_p_rem)
