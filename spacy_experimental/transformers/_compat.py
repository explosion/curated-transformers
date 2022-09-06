try:
    import transformers

    has_hf_transformers = True
except ImportError:
    transformers = None
    has_hf_transformers = False