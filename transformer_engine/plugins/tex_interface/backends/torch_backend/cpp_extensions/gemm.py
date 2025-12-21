# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Optional, Tuple, Union
import torch

__all__ = [
    "general_gemm_torch",
]

# DType enum to torch.dtype mapping
# DType values from base.py (corrected to match C++ enum):
# kByte=0, kInt32=2, kFloat32=4, kFloat16=5, kBFloat16=6, kFloat8E4M3=7, kFloat8E5M2=8, kFloat4E2M1=10
_DTYPE_TO_TORCH = {
    0: torch.uint8,          # kByte
    2: torch.int32,          # kInt32
    4: torch.float32,        # kFloat32
    5: torch.float16,        # kFloat16
    6: torch.bfloat16,       # kBFloat16
    7: torch.float8_e4m3fn,  # kFloat8E4M3
    8: torch.float8_e5m2,    # kFloat8E5M2
    # 10: kFloat4E2M1 - no torch equivalent yet
}


def _convert_dtype(dtype: Union[int, torch.dtype, None]) -> Optional[torch.dtype]:
    """Convert DType enum or torch.dtype to torch.dtype."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, int):
        return _DTYPE_TO_TORCH.get(dtype, None)
    # Try to get .value if it's an enum
    if hasattr(dtype, 'value'):
        return _DTYPE_TO_TORCH.get(dtype.value, None)
    return None


def general_gemm_torch(
    A: torch.Tensor,
    transA: bool,
    B: torch.Tensor,
    transB: bool,
    D: Optional[torch.Tensor],
    quantizer: Any,
    output_dtype: Any,
    bias: Optional[torch.Tensor],
    bias_type: Any,
    gelu: bool,
    gelu_in: Optional[torch.Tensor],
    grad: bool,
    workspace: torch.Tensor,
    workspace_size: int,
    accumulate: bool,
    use_split_accumulator: bool,
    comm_overlap: Optional[Any] = None,
    comm_type: Optional[Any] = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
    alpha: float = 1.0,
    beta: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    GEMM implementation using PyTorch native operations.

    TransformerEngine GEMM convention:
        out = B @ op(A)  (row-major convention)
        where op(A) = A.T if transA else A

    For Linear layer:
        - Forward (TN): y = input @ weight.T
        - dgrad (NN):   dx = dy @ weight
        - wgrad (NT):   dw = dy.T @ x

    Args:
        A: First input tensor (typically weight)
        transA: Whether to transpose A
        B: Second input tensor (typically activation)
        transB: Whether to transpose B
        D: Output tensor (can be None, will be created)
        quantizer: Quantization parameters (not used in torch backend)
        output_dtype: Output data type (DType enum or torch.dtype)
        bias: Optional bias tensor to add to output
        bias_type: Bias data type (not used in torch backend)
        gelu: Whether to apply GELU activation after bias
        gelu_in: Pre-allocated tensor to store pre-GELU values (for backward)
        grad: Whether this is a gradient computation
        workspace: Workspace tensor
        workspace_size: Size of workspace
        accumulate: Whether to accumulate into D (out = alpha*matmul + beta*D)
        use_split_accumulator: Whether to use split accumulator
        comm_overlap: Communication overlap object (not used in torch backend)
        comm_type: Communication type (not used in torch backend)
        extra_output: Extra output tensor (not used in torch backend)
        bulk_overlap: Whether to use bulk overlap (not used in torch backend)
        alpha: Scaling factor for matmul result
        beta: Scaling factor for D when accumulating

    Returns:
        Tuple of (output, bias_grad, gelu_input, extra_output)
    """
    import torch.nn.functional as F

    # Use B's device as the target device (B is the input activation tensor)
    target_device = B.device

    # Move A to B's device if they differ (A is typically the weight tensor)
    if A.device != target_device:
        A = A.to(target_device)

    # Store original shape for 3D tensors
    original_B_shape = None
    if B.ndim == 3:
        original_B_shape = B.shape
        B = B.reshape(-1, B.shape[-1])

    if A.ndim == 3:
        A = A.reshape(-1, A.shape[-1])

    # Apply transpositions according to layout
    # TransformerEngine convention: out = op(B) @ op(A)
    A_comp = A.T if transA else A
    B_comp = B.T if transB else B

    # Perform matrix multiplication
    # Keep original dtype for Tensor Core acceleration (FP16/BF16)
    # Only upcast FP8 types to their compute dtype
    if A_comp.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 needs upcast to compute dtype
        compute_dtype = torch.bfloat16
        A_comp = A_comp.to(compute_dtype)
        B_comp = B_comp.to(compute_dtype)

    out = torch.mm(B_comp, A_comp)

    # Apply alpha scaling
    if alpha != 1.0:
        out = out * alpha

    # Reshape output if B was 3D
    if original_B_shape is not None:
        out = out.view(original_B_shape[0], original_B_shape[1], -1)

    # Store pre-GELU input if needed (for backward pass)
    gelu_input_ret = None
    if gelu and gelu_in is not None:
        # Will store the pre-GELU values after bias addition
        pass

    # Add bias if provided
    if bias is not None:
        if bias.device != target_device:
            bias = bias.to(target_device)
        out = out + bias

    # Store pre-GELU values if GELU will be applied
    if gelu:
        if gelu_in is not None:
            gelu_in.copy_(out)
            gelu_input_ret = gelu_in
        else:
            gelu_input_ret = out.clone()
        # Apply GELU activation (tanh approximation as used in TransformerEngine)
        out = F.gelu(out, approximate='tanh')

    # Convert to output dtype if needed
    torch_out_dtype = _convert_dtype(output_dtype)
    if torch_out_dtype is not None and out.dtype != torch_out_dtype:
        out = out.to(torch_out_dtype)

    # Handle output tensor D
    if D is not None:
        if D.device != target_device:
            D = D.to(target_device)
        if accumulate:
            # out = alpha * matmul + beta * D
            beta_val = beta if beta is not None else 1.0
            D.mul_(beta_val).add_(out)
            out = D
        else:
            # Just copy result to D
            D.copy_(out)
            out = D

    # Compute bias gradient if this is a backward pass and bias was used
    bias_grad = None
    if grad and bias is not None:
        # In backward pass, bias_grad = sum of gradients over batch dimension
        # This is typically handled separately, so we leave it as None here
        pass

    extra_output_ret = None

    return out, bias_grad, gelu_input_ret, extra_output_ret
