from typing import Any, Tuple, Type, TypeVar, cast
import json

from .bbpe_tokenizer import ByteBPETokenizer
from .hf_hub import FromHFHub, FromPretrainedHFTokenizer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="GPTNeoXTokenizer")


class GPTNeoXTokenizer(ByteBPETokenizer, FromHFHub, FromPretrainedHFTokenizer):
    """GPT-NeoX tokenizer (Black et al., 2022).

    The GPT-NeoX tokenizer uses byte-level byte pair encoding.
    """

    @classmethod
    def _convert_hf_tokenizer_json(cls: Type[Self], *, hf_tokenizer: Any) -> Self:
        # TODO: Seems like we can't easily probe the model type?
        model = hf_tokenizer["model"]
        vocab = model["vocab"]
        merges = [
            cast(Tuple[str, str], tuple(merge.split(" ", maxsplit=2)))
            for merge in model["merges"]
        ]
        added_tokens = {
            added["content"]: added["id"] for added in hf_tokenizer["added_tokens"]
        }
        return cls(vocab=vocab, merges=merges, added_tokens=added_tokens)

    @classmethod
    def _convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        serialized = tokenizer.backend_tokenizer.to_str(True)  # type: ignore
        deserialized = json.loads(serialized)
        return cls._convert_hf_tokenizer_json(hf_tokenizer=deserialized)
