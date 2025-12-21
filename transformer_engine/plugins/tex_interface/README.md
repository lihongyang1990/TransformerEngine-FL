# TEX Interface - Unified Backend System for Transformer Engine

## Overview

TEX Interface provides a unified backend system that allows different hardware vendors to implement the `transformer_engine_torch` interface. This enables a simple one-line import change to switch between different hardware backends.

## Quick Start

### For Users

Original code:
```python
import transformer_engine_torch as tex
tex.rmsnorm_fwd(input, weight, eps, ...)
```

With TEX Interface (just change the import):
```python
# Option 1: Use the plugins.tex alias
from transformer_engine.plugins import tex
tex.rmsnorm_fwd(input, weight, eps, ...)

# Option 2: Use the full module path
from transformer_engine.plugins.transformer_engine_fl_torch import *
# Then tex.* will work the same way
```

### Backend Selection

By default, the best available backend is auto-selected. You can override this:

```bash
# Force NVIDIA backend
export TE_BACKEND=nvidia

# Force a specific vendor backend
export TE_BACKEND=hygon
```

Or programmatically:
```python
import transformer_engine_fl_torch as tex
tex.set_backend("hygon")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
│                                                              │
│   import transformer_engine_fl_torch as tex                  │
│   tex.rmsnorm_fwd(...)                                       │
├─────────────────────────────────────────────────────────────┤
│                  transformer_engine_fl_torch                 │
│                     (Module Entry Point)                     │
├─────────────────────────────────────────────────────────────┤
│                    TEXBackendBase                            │
│              (Abstract Interface Definition)                 │
│                                                              │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│   │ Quantize │ │   GEMM   │ │   Norm   │ │ Attention│ ...  │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
├───────────────┬───────────────┬───────────────┬─────────────┤
│    NVIDIA     │     Hygon     │      AMD      │   Others    │
│    Backend    │    Backend    │    Backend    │             │
├───────────────┼───────────────┼───────────────┼─────────────┤
│ transformer_  │   hygon_ops   │   rocm_ops    │    ...      │
│ engine_torch  │      .so      │      .so      │             │
└───────────────┴───────────────┴───────────────┴─────────────┘
```

## Directory Structure

```
transformer_engine/plugins/tex_interface/
├── __init__.py           # Package exports
├── base.py               # TEXBackendBase abstract class
├── registry.py           # Backend registration and selection
├── README.md             # This file
└── backends/
    ├── __init__.py       # Import backends to register
    ├── nvidia/           # NVIDIA backend (wraps transformer_engine_torch)
    │   ├── __init__.py
    │   └── nvidia.py
    ├── flaggems/         # FlagGems cross-platform backend (highest priority)
    │   ├── __init__.py
    │   ├── flaggems.py
    │   └── cpp_extensions/
    ├── torch_backend/    # Pure PyTorch fallback backend
    │   ├── __init__.py
    │   ├── torch_backend.py
    │   └── cpp_extensions/
    └── template.py       # Template for vendors

transformer_engine_fl_torch/
└── __init__.py           # Module entry point (drop-in replacement for tex)
```

## For Hardware Vendors

### Creating a New Backend

1. **Copy the template**:
   ```bash
   cp transformer_engine/plugins/tex_interface/backends/template.py \
      transformer_engine/plugins/tex_interface/backends/hygon.py
   ```

2. **Modify the backend class**:
   ```python
   from ..registry import register_backend

   @register_backend
   class HygonBackend(TEXBackendBase):
       NAME = "hygon"
       PRIORITY = 90  # High priority for Hygon hardware

       def is_available(self) -> bool:
           return check_hygon_available()

       def rmsnorm_fwd(self, input, weight, eps, ...):
           # Call your Hygon-specific kernel
           return hygon_rmsnorm_fwd(input, weight, eps, ...)
   ```

3. **Register your backend** in `backends/__init__.py`:
   ```python
   try:
       from . import hygon
   except ImportError:
       pass
   ```

### Implementation Priority

Methods are categorized by importance:

**CRITICAL (Must Optimize)**:
- `generic_gemm` - Matrix multiplication
- `rmsnorm_fwd/bwd` - RMSNorm operations
- `fused_attn_fwd/bwd` - Fused attention (if available)

**IMPORTANT**:
- `layernorm_fwd/bwd` - LayerNorm operations
- Activation functions (gelu, swiglu, etc.)
- Softmax operations

**OPTIONAL**:
- Communication overlap operations
- NVSHMEM operations (NVIDIA-specific)
- Advanced FP8 features

### Example: Hygon Backend

```python
@register_backend
class HygonBackend(TEXBackendBase):
    NAME = "hygon"
    PRIORITY = 90

    def __init__(self):
        self._hygon_lib = None

    def _get_lib(self):
        if self._hygon_lib is None:
            import hygon_te_ops
            self._hygon_lib = hygon_te_ops
        return self._hygon_lib

    @property
    def name(self) -> str:
        return "hygon"

    @property
    def vendor(self) -> str:
        return "Hygon"

    @property
    def priority(self) -> int:
        return 90

    def is_available(self) -> bool:
        try:
            import torch
            import hygon_te_ops
            return torch.hip.is_available()
        except ImportError:
            return False

    def rmsnorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        lib = self._get_lib()
        return lib.rmsnorm_fwd(
            input, weight, eps, ln_out, quantizer,
            otype, sm_margin, zero_centered_gamma
        )

    # ... implement other methods
```

## API Reference

### Backend Management

```python
from transformer_engine_fl_torch import get_backend, set_backend, list_backends

# List available backends
backends = list_backends()
# [{'name': 'nvidia', 'vendor': 'NVIDIA', 'priority': 100, 'available': True}, ...]

# Get current backend
backend = get_backend()

# Set specific backend
set_backend("hygon")
```

### TEXBackendBase Interface

The complete interface is defined in `base.py`. Key method categories:

#### Quantization
- `quantize(tensor, quantizer, output, noop)`
- `dequantize(input, otype)`
- `bgrad_quantize(input, quantizer)`

#### GEMM
- `generic_gemm(...)`
- `te_general_grouped_gemm(...)`

#### Normalization
- `rmsnorm_fwd(input, weight, eps, ...)`
- `rmsnorm_bwd(dy, x, rsigma, gamma, ...)`
- `layernorm_fwd(input, weight, bias, eps, ...)`
- `layernorm_bwd(dy, x, mu, rsigma, gamma, ...)`

#### Attention
- `fused_attn_fwd(...)`
- `fused_attn_bwd(...)`
- `get_fused_attn_backend(...)`

#### Activations
- `gelu`, `geglu`, `qgelu`, `relu`, `silu`, `swiglu`, ...
- `dgelu`, `dgeglu`, `dqgelu`, `drelu`, `dsilu`, `dswiglu`, ...

#### Optimizers
- `multi_tensor_adam(...)`
- `multi_tensor_sgd(...)`
- `multi_tensor_scale(...)`
- `multi_tensor_l2norm(...)`

## Testing Your Backend

```python
import transformer_engine_fl_torch as tex
import torch

# Set your backend
tex.set_backend("your_vendor")

# Test basic operations
x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)

# Test RMSNorm
output, _, rsigma = tex.rmsnorm_fwd(
    x, weight, 1e-5, None, None, torch.bfloat16, 0, False
)
print(f"RMSNorm output shape: {output.shape}")

# Test GEMM
A = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
B = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
D = torch.empty(1024, 2048, device="cuda", dtype=torch.bfloat16)
workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)

result = tex.generic_gemm(
    A, False, B, False, D, None, torch.bfloat16,
    None, None, False, None, False, workspace, 1024*1024,
    False, False
)
print(f"GEMM output shape: {D.shape}")
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TE_BACKEND` | Force specific backend | `TE_BACKEND=torch` |
| `TE_PLUGIN` | Alternative to TE_BACKEND | `TE_PLUGIN=nvidia` |
| `TE_FL_DEBUG` | Enable debug printing | `TE_FL_DEBUG=1` |
| `TE_FL_SKIP_CUDA` | Disable NVIDIA backend | `TE_FL_SKIP_CUDA=1` |
| `TE_FALLBACK_BACKEND` | FlagGems fallback backend | `TE_FALLBACK_BACKEND=torch` |

## Troubleshooting

### Backend Not Found
```
RuntimeError: Backend 'hygon' not found. Available backends: ['nvidia']
```
- Ensure your backend is registered in `backends/__init__.py`
- Check that your backend's `is_available()` returns `True`

### Import Error
```
ImportError: cannot import name 'HygonBackend'
```
- Check that all dependencies are installed
- Verify the backend file has no syntax errors

### Method Not Implemented
```
NotImplementedError: fused_attn_fwd not implemented
```
- This method is not yet implemented in your backend
- Either implement it or the calling code will need to use a fallback

## Contributing

1. Fork the repository
2. Create your backend in `backends/your_vendor.py`
3. Add comprehensive tests
4. Submit a pull request

## License

See LICENSE file in the repository root.
