# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

# GEMM operations
from .gemm import general_gemm_torch

# Normalization operations
from .rmsnorm import rmsnorm_fwd_torch, rmsnorm_bwd_torch
from .normalization import layernorm_fwd_torch, layernorm_bwd_torch

# Activation functions
from .activation import (
    # Forward
    gelu_torch, geglu_torch, qgelu_torch, qgeglu_torch,
    relu_torch, reglu_torch, srelu_torch, sreglu_torch,
    silu_torch, swiglu_torch, clamped_swiglu_torch,
    # Backward
    dgelu_torch, dgeglu_torch, dqgelu_torch, dqgeglu_torch,
    drelu_torch, dreglu_torch, dsrelu_torch, dsreglu_torch,
    dsilu_torch, dswiglu_torch, clamped_dswiglu_torch,
    # Fused
    dbias_dgelu_torch, dbias_dsilu_torch, dbias_drelu_torch,
    dbias_dqgelu_torch, dbias_dsrelu_torch,
)

# Softmax operations
from .softmax import (
    scaled_softmax_forward_torch,
    scaled_softmax_backward_torch,
    scaled_masked_softmax_forward_torch,
    scaled_masked_softmax_backward_torch,
    scaled_upper_triang_masked_softmax_forward_torch,
    scaled_upper_triang_masked_softmax_backward_torch,
    scaled_aligned_causal_masked_softmax_forward_torch,
    scaled_aligned_causal_masked_softmax_backward_torch,
)

# Dropout operations
from .dropout import dropout_fwd_torch, dropout_bwd_torch

# Optimizer operations
from .optimizer import (
    multi_tensor_scale_torch,
    multi_tensor_l2norm_torch,
    multi_tensor_adam_torch,
    multi_tensor_sgd_torch,
)

__all__ = [
    # GEMM
    "general_gemm_torch",
    # Normalization
    "rmsnorm_fwd_torch",
    "rmsnorm_bwd_torch",
    "layernorm_fwd_torch",
    "layernorm_bwd_torch",
    # Activation - Forward
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
    # Activation - Backward
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
    # Activation - Fused
    "dbias_dgelu_torch",
    "dbias_dsilu_torch",
    "dbias_drelu_torch",
    "dbias_dqgelu_torch",
    "dbias_dsrelu_torch",
    # Softmax
    "scaled_softmax_forward_torch",
    "scaled_softmax_backward_torch",
    "scaled_masked_softmax_forward_torch",
    "scaled_masked_softmax_backward_torch",
    "scaled_upper_triang_masked_softmax_forward_torch",
    "scaled_upper_triang_masked_softmax_backward_torch",
    "scaled_aligned_causal_masked_softmax_forward_torch",
    "scaled_aligned_causal_masked_softmax_backward_torch",
    # Dropout
    "dropout_fwd_torch",
    "dropout_bwd_torch",
    # Optimizer
    "multi_tensor_scale_torch",
    "multi_tensor_l2norm_torch",
    "multi_tensor_adam_torch",
    "multi_tensor_sgd_torch",
]
