try:
    import transformers

    has_hf_transformers = True
except ImportError:
    transformers = None  # type: ignore
    has_hf_transformers = False

try:
    import bitsandbytes

    has_bitsandbytes = True

    # Check if we have the correct version (that exposes the `device` parameter).
    try:
        _ = bitsandbytes.nn.Linear8bitLt(1, 1, device=None)
        _ = bitsandbytes.nn.Linear4bit(1, 1, device=None)
        has_bitsandbytes_linear_device = True
    except TypeError:
        has_bitsandbytes_linear_device = False
except ImportError:
    bitsandbytes = None  # type: ignore
    has_bitsandbytes = False
    has_bitsandbytes_linear_device = False
