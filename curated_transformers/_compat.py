try:
    import transformers

    has_hf_transformers = True
except ImportError:
    transformers = None
    has_hf_transformers = False

try:
    import huggingface_hub

    has_huggingface_hub = True
except ImportError:
    huggingface_hub = None
    has_huggingface_hub = False
