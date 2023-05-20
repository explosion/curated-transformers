from typing import Iterable, List
from cutlery import SentencePieceProcessor

from .tokenizer import PiecesWithIds, Tokenizer


class SentencePieceTokenizer(Tokenizer):
    """Piece tokenizer using SentencePiece encoding
    (Kudo et al., 2018)"""

    _FAIRSEQ_BOS = 0
    _FAIRSEQ_EOS = 2
    _FAIRSEQ_UNK = 3

    _SPP_BOS = 1
    _SPP_EOS = 2
    _SPP_UNK = 0

    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
    ):
        """Construct a tokenizer from a cutlery SentencePiece processor.

        processor (SentencePieceProcessor): The processor to wrap.
        """
        self.processor = processor

    def _decode(self, input: Iterable[Iterable[int]]) -> List[str]:
        return [self.processor.decode_from_ids(ids) for ids in input]

    def _encode(self, input: Iterable[str]) -> PiecesWithIds:
        ids = []
        pieces = []
        lens = []

        for text in input:
            text_ids = []
            text_pieces = []
            text_lens = []

            for token in text.split(" "):
                token_ids, token_pieces = self.processor.encode(token)
                text_ids.extend(token_ids)
                text_pieces.extend(token_pieces)
                text_lens.append(len(token_ids))

            ids.append(text_ids)
            pieces.append(text_pieces)
            lens.append(text_lens)

        return PiecesWithIds(ids=ids, lens=lens, pieces=pieces)
