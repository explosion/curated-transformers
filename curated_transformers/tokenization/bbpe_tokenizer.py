from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, TypeVar, cast
from curated_tokenizers import ByteBPEProcessor
import json

from .chunks import (
    MergedInputChunks,
    MergedSpecialPieceChunk,
)
from .hf_hub import (
    FromHFHub,
    FromPretrainedHFTokenizer,
)
from .tokenizer import PiecesWithIds, Tokenizer


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="ByteBPETokenizer")


class ByteBPETokenizer(Tokenizer, FromHFHub, FromPretrainedHFTokenizer):
    """
    Piece tokenizer using byte-level byte pair encoding
    (Gage, 1994, Sennrich et al., 2016)
    """

    def __init__(
        self,
        *,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        special_pieces: Optional[Dict[str, int]] = None,
    ):
        """
        Construct a byte BPE tokenizer.

        :param vocab:
            The word piece vocabulary.
        :param merges:
            Merges.
        :param special_pieces:
            Special pieces.
        """
        self.special_piece_to_id = {} if special_pieces is None else special_pieces
        self.id_to_special_piece = {v: k for k, v in self.special_piece_to_id.items()}
        vocab.update(self.special_piece_to_id)
        self.processor = ByteBPEProcessor(vocab, merges)

    def _decode(
        self, input: Iterable[Iterable[int]], skip_special_pieces: bool
    ) -> List[str]:
        input = [
            [
                id
                for id in ids
                if not skip_special_pieces or id not in self.id_to_special_piece
            ]
            for ids in input
        ]
        return [self.processor.decode_from_ids(ids) for ids in input]

    def _encode(self, input: Iterable[MergedInputChunks]) -> PiecesWithIds:
        ids = []
        pieces = []

        for seq in input:
            seq_ids = []
            seq_pieces = []

            for chunk in seq:
                if isinstance(chunk, MergedSpecialPieceChunk):
                    piece_id = self.processor.piece_id(chunk.piece)
                    if piece_id is None:
                        raise ValueError(f"Unknown special piece: {chunk.piece}")
                    seq_ids.append(piece_id)
                    seq_pieces.append(chunk.piece)
                else:
                    for idx, token in enumerate(chunk.text.split(" ")):
                        if idx != 0:
                            token = " " + token
                        token_ids, token_pieces = self.processor.encode(token)
                        seq_ids.extend(token_ids)
                        seq_pieces.extend(token_pieces)

            ids.append(seq_ids)
            pieces.append(seq_pieces)

        return PiecesWithIds(ids=ids, pieces=pieces)

    def _special_tokens(self) -> Set[str]:
        return set(self.special_piece_to_id.keys())

    @classmethod
    def _convert_hf_tokenizer_json(
        cls: Type[Self], *, hf_tokenizer: Dict[str, Any]
    ) -> Self:
        model = hf_tokenizer["model"]
        if model["type"] != "BPE":
            raise ValueError(
                "Attempted to load a non-Byte BPE tokenizer as a Byte BPE tokenizer"
            )

        vocab = model["vocab"]
        merges = [
            cast(Tuple[str, str], tuple(merge.split(" ", maxsplit=2)))
            for merge in model["merges"]
        ]
        special_pieces = {
            added["content"]: added["id"] for added in hf_tokenizer["added_tokens"]
        }
        return cls(vocab=vocab, merges=merges, special_pieces=special_pieces)

    @classmethod
    def _convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        serialized = tokenizer.backend_tokenizer.to_str(True)  # type: ignore
        deserialized = json.loads(serialized)
        return cls._convert_hf_tokenizer_json(hf_tokenizer=deserialized)
