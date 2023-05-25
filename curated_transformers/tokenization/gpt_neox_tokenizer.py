from .bbpe_tokenizer import ByteBPETokenizer


class GPTNeoXTokenizer(ByteBPETokenizer):
    """GPT-NeoX tokenizer (Black et al., 2022).

    The GPT-NeoX tokenizer uses byte-level byte pair encoding.
    """
