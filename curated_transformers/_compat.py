from spacy.lang.ja import try_sudachi_import

try:
    import transformers

    has_hf_transformers = True
except ImportError:
    transformers = None  # type: ignore
    has_hf_transformers = False

try:
    import huggingface_hub

    has_huggingface_hub = True
except ImportError:
    huggingface_hub = None  # type: ignore
    has_huggingface_hub = False


try:
    import fugashi

    has_fugashi = True
except ImportError:
    fugashi = None
    has_fugashi = False

try:
    try_sudachi_import()
    has_sudachi = True
except ImportError:
    has_sudachi = False
