from typing import Iterable, List
from cutlery import SentencePieceProcessor

from .tokenizer import PiecesWithIds, Tokenizer


class SentencePieceTokenizer(Tokenizer):
    """Piece tokenizer using SentencePiece encoding
    (Kudo et al., 2018)"""

    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
    ):
        """Construct a tokenizer from a cutlery SentencePiece processor.

        :param processor: The processor to wrap.
        """
        self.processor = processor

    def _decode(self, input: Iterable[Iterable[int]]) -> List[str]:
        return [self.processor.decode_from_ids(ids) for ids in input]

    def _encode(self, input: Iterable[str]) -> PiecesWithIds:
        ids = []
        pieces = []
        lens = []

        for text in input:
            text_lens = []

            text_ids, text_pieces = self.processor.encode(text)
            text_lens.append(len(text_ids))

            ids.append(text_ids)
            pieces.append(text_pieces)
            lens.append(text_lens)

        return PiecesWithIds(ids=ids, lens=lens, pieces=pieces)
