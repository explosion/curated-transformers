from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, TypeVar, cast
from curated_tokenizers import ByteBPEProcessor
import json
from pathlib import Path

from .bbpe_tokenizer import ByteBPETokenizer
from .tokenizer import AddBosEosPreEncoder, PreDecoder
from .util import remove_pieces_from_sequence


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="RobertaTokenizer")


class RobertaPreDecoder(PreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
    ):
        """Construct a RoBERTa pre-decoder.

        bos_id (int): The piece id used to mark the beginning of a sequence.
        eos_id (int): The piece id used to mark the end of a sequence.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, input: Iterable[Iterable[int]]) -> List[List[int]]:
        return [
            list(remove_pieces_from_sequence(ids, (self.bos_id, self.eos_id)))
            for ids in input
        ]


class RobertaTokenizer(ByteBPETokenizer):
    def __init__(
        self,
        *,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        added_tokens: Optional[Dict[str, int]] = None,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ):
        """
        Construct a RoBERTa tokenizer.

        :param vocab:
            The word piece vocabulary.
        :param merges:
            Merges.
        :param added_tokens:
            Additional tokens.
        :param bos_piece:
            Beginning of sequence piece.
        :param eos_piece:
            End of sequence piece.
        """
        super().__init__(vocab=vocab, merges=merges, added_tokens=added_tokens)

        self.bos_piece = bos_piece
        self.eos_piece = eos_piece

        bos_id = _get_piece_id_or_fail(self.processor, bos_piece)
        eos_id = _get_piece_id_or_fail(self.processor, eos_piece)

        self.pre_decoder = RobertaPreDecoder(
            bos_id=bos_id,
            eos_id=eos_id,
        )

        self.pre_encoder = AddBosEosPreEncoder(bos_piece=bos_piece, eos_piece=eos_piece)

    def _special_tokens(self) -> Set[str]:
        special_tokens = {self.bos_piece, self.eos_piece}
        special_tokens.update(super()._special_tokens())
        return special_tokens

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        vocab_path: Path,
        merges_path: Path,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ) -> Self:
        """Construct a tokenizer from vocabulary and merge files.

        vocab_path (Path): path to the vocabulary file.
        merges_path (Path): path to the merges file.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
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

    @classmethod
    def _convert_hf_tokenizer_json(
        cls: Type[Self], *, hf_tokenizer: Dict[str, Any]
    ) -> Self:
        if hf_tokenizer["post_processor"]["type"] != "RobertaProcessing":
            raise ValueError(
                "Attempted to load a non-RoBERTa tokenizer as a RoBERTa tokenizer"
            )

        model = hf_tokenizer["model"]
        vocab = model["vocab"]
        merges = [
            cast(Tuple[str, str], tuple(merge.split(" ", maxsplit=2)))
            for merge in model["merges"]
        ]
        added_tokens = {
            added["content"]: added["id"] for added in hf_tokenizer["added_tokens"]
        }

        post_processor = hf_tokenizer["post_processor"]
        bos_piece = post_processor["cls"][0]
        eos_piece = post_processor["sep"][0]

        return cls(
            vocab=vocab,
            merges=merges,
            added_tokens=added_tokens,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
        )

    @classmethod
    def _convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        serialized = tokenizer.backend_tokenizer.to_str(True)  # type: ignore
        deserialized = json.loads(serialized)
        return cls._convert_hf_tokenizer_json(hf_tokenizer=deserialized)


def _get_piece_id_or_fail(processor: ByteBPEProcessor, piece: str):
    piece_id = processor.piece_id(piece)
    if piece_id is None:
        raise ValueError(
            f"RoBERTa piece encoder vocabulary doesn't contain '{piece}' piece"
        )
    return piece_id
