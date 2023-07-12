from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from curated_tokenizers import ByteBPEProcessor

from ..hf_hub import LegacyFromHFHub
from ..util import remove_pieces_from_sequence
from .bbpe_tokenizer import ByteBPETokenizer
from .legacy_tokenizer import AddBosEosPreEncoder, PreDecoder

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="RoBERTaTokenizer")


class RoBERTaPreDecoder(PreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
    ):
        """
        Construct a RoBERTa pre-decoder.

        :param bos_id:
            The piece id used to mark the beginning of a sequence.
        :param eos_id:
            The piece id used to mark the end of a sequence.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, input: Iterable[Iterable[int]]) -> List[List[int]]:
        return [
            list(remove_pieces_from_sequence(ids, (self.bos_id, self.eos_id)))
            for ids in input
        ]


class RoBERTaTokenizer(ByteBPETokenizer, LegacyFromHFHub):
    """
    Legacy tokenizer for RoBERTa (`Liu et al., 2019`_) models.

    .. _Liu et al., 2019: https://arxiv.org/abs/1907.11692
    """

    vocab_files: Dict[str, str] = {"vocab": "vocab.json", "merges": "merges.txt"}

    def __init__(
        self,
        *,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        special_pieces: Optional[Dict[str, int]] = None,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ):
        """
        Construct a RoBERTa tokenizer.

        :param vocab:
            The word piece vocabulary.
        :param merges:
            Merges.
        :param special_pieces:
            Special pieces.
        :param bos_piece:
            Beginning of sequence piece.
        :param eos_piece:
            End of sequence piece.
        """
        super().__init__(vocab=vocab, merges=merges, special_pieces=special_pieces)

        bos_id = _get_piece_id_or_fail(self.processor, bos_piece)
        eos_id = _get_piece_id_or_fail(self.processor, eos_piece)

        self._eos_piece = eos_piece

        self.pre_decoder = RoBERTaPreDecoder(
            bos_id=bos_id,
            eos_id=eos_id,
        )

        self.pre_encoder = AddBosEosPreEncoder(bos_piece=bos_piece, eos_piece=eos_piece)

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        vocab_path: Path,
        merges_path: Path,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ) -> Self:
        """
        Construct a tokenizer from vocabulary and merge files.

        :param vocab_path:
            Path to the vocabulary file.
        :param merges_path:
            Path to the merges file.
        :param bos_piece:
            The piece to use to mark the beginning of a sequence.
        :param eos_piece:
            The piece to use to mark the end of a sequence.
        """
        processor = ByteBPEProcessor.load_from_files(
            vocab=vocab_path, merges=merges_path
        )
        return cls(
            # This is a bit annoying, but we want to avoid these extremely
            # overloaded constructors.
            vocab=processor.vocab,
            merges=processor.merges,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
        )

    @property
    def eos_piece(self) -> Optional[str]:
        return self._eos_piece

    @classmethod
    def _load_from_vocab_files(
        cls: Type[Self],
        *,
        vocab_files: Dict[str, Path],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> Self:
        return cls.from_files(
            vocab_path=vocab_files["vocab"], merges_path=vocab_files["merges"]
        )


def _get_piece_id_or_fail(processor: ByteBPEProcessor, piece: str):
    piece_id = processor.piece_to_id(piece)
    if piece_id is None:
        raise ValueError(
            f"RoBERTa piece encoder vocabulary doesn't contain '{piece}' piece"
        )
    return piece_id
