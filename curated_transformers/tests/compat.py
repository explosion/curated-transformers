import torch
from torch.nn import Identity

try:
    import transformers

    has_hf_transformers = True
except ImportError:
    transformers = None  # type: ignore
    has_hf_transformers = False

has_torch_compile = hasattr(torch, "compile")
if has_torch_compile:
    # torch.compile is not supported on all Python versions and platforms.
    try:
        torch.compile(Identity(), disable=True)
    except:
        has_torch_compile = False

try:
    import safetensors

    has_safetensors = True
except ImportError:
    safetensors = None  # type: ignore
    has_safetensors = False
