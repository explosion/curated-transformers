from typing import Any, Type, TypeVar
from curated_tokenizers import ByteBPEProcessor
import json
from pathlib import Path

from .bbpe_tokenizer import ByteBPETokenizer
from .hf_hub import FromPretrainedHFTokenizer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="GPTNeoXTokenizer")


class GPTNeoXTokenizer(ByteBPETokenizer, FromPretrainedHFTokenizer):
    """GPT-NeoX tokenizer (Black et al., 2022).

    The GPT-NeoX tokenizer uses byte-level byte pair encoding.
    """

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        vocab_path: Path,
        merges_path: Path,
    ) -> Self:
        """Construct a tokenizer from the vocabulary and merges files.

        vocab_path (Path): path to the vocabulary file.
        merges_path (Path): path to the merges file.
        """
        processor = ByteBPEProcessor.load_from_files(
            vocab=vocab_path, merges=merges_path
        )
        return cls(processor=processor)

    @classmethod
    def convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        serialized = tokenizer.backend_tokenizer.to_str(True)  # type: ignore
        deserialized = json.loads(serialized)
        vocab_merges = deserialized["model"]
        merges = [tuple(merge.split(" ")) for merge in vocab_merges["merges"]]
        processor = ByteBPEProcessor(vocab_merges["vocab"], merges)
        return cls(
            processor=processor,
        )
