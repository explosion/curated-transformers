from typing import Optional

from huggingface_hub import HfFileSystem

from ..compat import transformers
from ..utils import torch_assertclose


def compare_tokenizer_outputs_with_hf_tokenizer(
    sample_texts,
    hf_name,
    tokenizer_cls,
    pad_token: Optional[str] = None,
    with_hf_fast: bool = True,
    with_fsspec: bool = False,
    revision: str = "main",
):
    if with_fsspec:
        tokenizer = tokenizer_cls.from_fsspec(fs=HfFileSystem(), model_path=hf_name)
    else:
        tokenizer = tokenizer_cls.from_hf_hub(name=hf_name, revision=revision)
    pieces = tokenizer(sample_texts)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_name, revision=revision, use_fast=with_hf_fast
    )
    hf_tokenizer.padding_side = "right"
    if pad_token is not None:
        hf_tokenizer.add_special_tokens({"pad_token": pad_token})

    # Test encoding with right-padding.
    hf_pieces = hf_tokenizer(sample_texts, padding=True, return_tensors="pt")
    torch_assertclose(
        pieces.padded_tensor(padding_id=hf_tokenizer.pad_token_id),
        hf_pieces["input_ids"].int(),
    )
    torch_assertclose(
        pieces.attention_mask().bool_mask.squeeze(dim=(1, 2)),
        hf_pieces["attention_mask"].bool(),
    )

    # Test decoding
    decoded = tokenizer.decode(pieces.ids)
    hf_decoded = [
        hf_tokenizer.decode(ids, skip_special_tokens=True)
        for ids in hf_pieces["input_ids"]
    ]

    assert decoded == hf_decoded

    # Test encoding with left-padding.
    hf_tokenizer.padding_side = "left"
    hf_pieces = hf_tokenizer(sample_texts, padding=True, return_tensors="pt")
    torch_assertclose(
        pieces.padded_tensor(padding_id=hf_tokenizer.pad_token_id, pad_left=True),
        hf_pieces["input_ids"].int(),
    )
    torch_assertclose(
        pieces.attention_mask(pad_left=True).bool_mask.squeeze(dim=(1, 2)),
        hf_pieces["attention_mask"].bool(),
    )
