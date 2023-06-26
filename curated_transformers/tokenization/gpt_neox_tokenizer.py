from .bbpe_tokenizer import ByteBPETokenizer


class GPTNeoXTokenizer(ByteBPETokenizer):
    """
    GPT-NeoX (Black et al., 2022) tokenizer.

    The GPT-NeoX tokenizer uses byte-level byte pair encoding.
    """
