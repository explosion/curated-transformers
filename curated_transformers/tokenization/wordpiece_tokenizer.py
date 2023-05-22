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


def clean_up_decoded_string_like_hf(text: str) -> str:
    """This method is provided to ensure that the decoded string
    is compatible with the decoded output of the corresponding HF
    WordPiece tokenizer.

    c.f https://github.com/huggingface/tokenizers/blob/b4fcc9ce6e4ad5806e82826f816acfdfdc4fcc67/tokenizers/src/decoders/wordpiece.rs#L31"""
    cleaned_up = (
        text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return cleaned_up
