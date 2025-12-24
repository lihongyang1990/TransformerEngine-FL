# ACME TransformerEngine Plugin

This is an example out-of-tree vendor plugin for TransformerEngine-FL, demonstrating how to create and distribute a closed-source vendor implementation as a wheel package.

## Features

- 🚀 Optimized RMSNorm implementation
- 🔄 RoPE (Rotary Position Embedding) support
- ⚡ Flash Attention integration
- 📦 Easy installation via pip
- 🔌 Automatic plugin discovery
- ⚙️ Configurable via environment variables

## Installation

### From Source

```bash
# Clone or download this package
cd acme-te-plugin

# Install in development mode
pip install -e .

# Or build and install wheel
pip install build
python -m build
pip install dist/acme_te_plugin-1.0.0-py3-none-any.whl
```

### From PyPI (if published)

```bash
pip install acme-te-plugin
```

## Usage

Once installed, the plugin is automatically discovered and loaded by TransformerEngine-FL:

```python
import torch
import transformer_engine_fl as te_fl

# The plugin is automatically loaded!
# Use TE-FL operations normally

input_tensor = torch.randn(2, 512, 1024, device='cuda')
weight = torch.randn(1024, device='cuda')

# ACME's optimized implementation will be used automatically
output, rsigma = te_fl.rmsnorm_fwd(input_tensor, weight, eps=1e-5)
```

## Configuration

### Prefer ACME Vendor

```bash
# Ensure ACME vendor is preferred (default behavior)
export TE_FL_PREFER_VENDOR=1
```

### Force ACME Implementation

```bash
# Use only ACME vendor for specific operations
export TE_FL_PER_OP="rmsnorm_fwd=vendor:acme;rope_fwd=vendor:acme"
```

### Allow Only ACME

```bash
# Whitelist only ACME vendor
export TE_FL_ALLOW_VENDORS=acme
```

## Hardware Requirements

- ACME AI Accelerator (v2 or later)
- ACME Driver version 3.0+
- CUDA-compatible for fallback

## Supported Operations

| Operation | Priority | Supported Dtypes | Min Architecture |
|-----------|----------|------------------|------------------|
| rmsnorm_fwd | 150 | fp16, bf16, fp32 | acme_v2 |
| rope_fwd | 140 | fp16, bf16 | acme_v2 |
| flash_attention | 160 | fp16, bf16 | acme_v3 |

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Building Documentation

```bash
pip install sphinx sphinx-rtd-theme
cd docs
make html
```

## Troubleshooting

### Plugin Not Loaded

Check if the plugin is discovered:

```bash
export TE_FL_DEBUG=1
python -c "import transformer_engine_fl"
```

Look for:
```
[TEFL Discovery] Loading entry point: acme
[TEFL Discovery] Registered plugin from entry_point:acme
```

### ACME Device Not Available

The plugin includes availability checks. If ACME hardware is not available, it will gracefully fall back to other implementations:

```python
from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry
registry = OpRegistry()
# Check if ACME implementation is available
impls = registry.get_implementations("rmsnorm_fwd")
acme_impl = next(impl for impl in impls if impl.vendor == "acme")
print(f"ACME available: {acme_impl.is_available()}")
```

## License

Copyright (c) 2025 ACME Corporation. All rights reserved.

This is example code for demonstration purposes.

## Support

- Documentation: https://acme-te-plugin.readthedocs.io
- Issues: https://github.com/acme/acme-te-plugin/issues
- Email: support@acme.example.com

## See Also

- [TransformerEngine-FL](https://github.com/BAAI/TransformerEngine-FL)
- [ACME AI Accelerator](https://acme.example.com/products/ai-accelerator)
