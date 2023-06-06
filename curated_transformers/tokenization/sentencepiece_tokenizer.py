from typing import Iterable, List, Set
from curated_tokenizers import SentencePieceProcessor

from .chunks import MergedInputChunks, MergedSpecialPieceChunk
from .tokenizer import PiecesWithIds, Tokenizer


class SentencePieceTokenizer(Tokenizer):
    """Piece tokenizer using SentencePiece encoding
    (Kudo et al., 2018)"""

    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
    ):
        """Construct a tokenizer from a curated tokenizers SentencePiece processor.

        :param processor: The processor to wrap.
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
                    # TODO: this is not ideal. piece_id() should probably return
                    # None for unknown pieces.
                    unk_id = self.processor.unk_id()
                    unk_piece = self.processor.id_to_piece(unk_id)
                    piece_id = self.processor.piece_to_id(chunk.piece)
                    if piece_id == unk_id and chunk.piece != unk_piece:
                        raise ValueError(f"Unknown special piece: {chunk.piece}")
                    seq_ids.append(piece_id)
                    seq_pieces.append(chunk.piece)
                else:
                    chunk_ids, chunk_pieces = self.processor.encode(chunk.text)
                    seq_ids.extend(chunk_ids)
                    seq_pieces.extend(chunk_pieces)

            ids.append(seq_ids)
            pieces.append(seq_pieces)

        return PiecesWithIds(ids=ids, pieces=pieces)

    def _special_tokens(self) -> Set[str]:
        special_tokens = {
            self.processor.id_to_piece(self.processor.bos_id()),
            self.processor.id_to_piece(self.processor.eos_id()),
            self.processor.id_to_piece(self.processor.unk_id()),
        }

        return special_tokens
