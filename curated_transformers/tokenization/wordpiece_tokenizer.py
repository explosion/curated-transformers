from typing import Iterable, List
from cutlery import WordPieceProcessor

from .tokenizer import PiecesWithIds, Tokenizer


class WordPieceTokenizer(Tokenizer):
    """Piece tokenizer using WordPiece tokenization
    (Delvin et al., 2018)"""

    def __init__(
        self,
        *,
        processor: WordPieceProcessor,
    ):
        """Construct a tokenizer from a cutlery WordPiece processor.

        :param processor: The processor to wrap.
        """
        self.processor = processor

    def _decode(self, input: Iterable[Iterable[int]]) -> List[str]:
        decoded = []
        for piece_ids in input:
            tokens = []
            for piece_id in piece_ids:
                token, is_initial = self.processor.id_to_piece(piece_id)
                if is_initial:
                    token = " " + token
                tokens.append(token)
            decoded.append("".join(tokens))
        return decoded

    def _encode(self, input: Iterable[str]) -> PiecesWithIds:
        ids = []
        pieces = []
        lens = []

        for text in input:
            text_ids = []
            text_pieces = []
            text_lens = []

            # We expect all input texts to be whitespace-splittable at this
            # point. This includes punctuation.
            for token in text.split(" "):
                token_ids, token_pieces = self.processor.encode(token)
                text_ids.extend(token_ids)
                text_pieces.extend(token_pieces)
                text_lens.append(len(token_ids))

            ids.append(text_ids)
            pieces.append(text_pieces)
            lens.append(text_lens)

        return PiecesWithIds(ids=ids, lens=lens, pieces=pieces)
