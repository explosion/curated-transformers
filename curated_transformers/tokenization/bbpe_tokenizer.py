from typing import Iterable, List
from curated_tokenizers import ByteBPEProcessor

from curated_transformers.tokenization.chunks import (
    MergedInputChunks,
    MergedSpecialPieceChunk,
)

from .tokenizer import PiecesWithIds, Tokenizer


class ByteBPETokenizer(Tokenizer):
    """Piece tokenizer using byte-level byte pair encoding
    (Gage, 1994, Sennrich et al., 2016)"""

    def __init__(
        self,
        *,
        processor: ByteBPEProcessor,
    ):
        """Construct a tokenizer from a curated tokenizers byte-level BPE processor.

        processor (ByteBPEProcessor): The processor to wrap.
        """
        self.processor = processor

    def _decode(self, input: Iterable[Iterable[int]]) -> List[str]:
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
