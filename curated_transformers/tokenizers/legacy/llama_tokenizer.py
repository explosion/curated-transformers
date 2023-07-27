from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from curated_tokenizers import SentencePieceProcessor

from ..hf_hub import LegacyFromHFHub
from .legacy_tokenizer import AddBosEosPreEncoder
from .sentencepiece_tokenizer import SentencePieceTokenizer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="LLaMATokenizer")

DEFAULT_BOS_PIECE = "<s>"


class LLaMATokenizer(SentencePieceTokenizer, LegacyFromHFHub):
    """
    Legacy tokenizer for LLaMA (`Touvron et al., 2023 [a]`_, `Touvron et al., 2023 [b]`_) models.

    .. _Touvron et al., 2023 [a]: https://arxiv.org/abs/2302.13971
    .. _Touvron et al., 2023 [b]: https://arxiv.org/abs/2307.09288
    """

    vocab_files: Dict[str, str] = {"model": "tokenizer.model"}

    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
        add_bos_piece: bool = True,
        add_eos_piece: bool = False,
    ):
        """
        Construct a LLaMA tokenizer from a ``curated-tokenizers`` SentencePiece processor.

        :param processor:
            The processor to wrap.
        :param add_bos_piece:
            Add a begin-of-sequence piece.
        :param add_eos_piece:
            Add an end-of-sequence piece.
        """
        super().__init__(processor=processor)

        self.processor = processor

        bos_piece = processor.id_to_piece(processor.bos_id()) if add_bos_piece else None
        eos_piece = processor.id_to_piece(processor.eos_id()) if add_eos_piece else None

        self.pre_encoder = AddBosEosPreEncoder(bos_piece=bos_piece, eos_piece=eos_piece)

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        model_path: Path,
        add_bos_piece: bool = True,
        add_eos_piece: bool = False,
    ) -> Self:
        """
        Construct a LLaMA tokenizer from a SentencePiece model.

        :param model_path:
            Path to the SentencePiece model file.
        :param add_bos_piece:
            Add a begin-of-sequence piece.
        :param add_eos_piece:
            Add an end-of-sequence piece.
        """
        processor = SentencePieceProcessor.from_file(str(model_path))
        return cls(
            processor=processor,
            add_bos_piece=add_bos_piece,
            add_eos_piece=add_eos_piece,
        )

    @classmethod
    def _load_from_vocab_files(
        cls: Type[Self],
        *,
        vocab_files: Dict[str, Path],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> Self:
        if tokenizer_config is None:
            return cls.from_files(model_path=vocab_files["model"])

        add_bos_piece = tokenizer_config.get("add_bos_token", True)
        add_eos_piece = tokenizer_config.get("add_eos_token", False)

        return cls.from_files(
            model_path=vocab_files["model"],
            add_bos_piece=add_bos_piece,
            add_eos_piece=add_eos_piece,
        )
