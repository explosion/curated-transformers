try:
    import transformers

    has_hf_transformers = True
except ImportError:
    transformers = None  # type: ignore
    has_hf_transformers = False

try:
    import bitsandbytes

    has_bitsandbytes = True
except ImportError:
    bitsandbytes = None  # type: ignore
    has_bitsandbytes = False
