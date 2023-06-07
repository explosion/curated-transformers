from typing import Dict, Iterable, List, Optional
from curated_tokenizers import WordPieceProcessor

from .chunks import MergedInputChunks, MergedSpecialPieceChunk
from .tokenizer import PiecesWithIds, Tokenizer


class WordPieceTokenizer(Tokenizer):
    """Piece tokenizer using WordPiece tokenization
    (Delvin et al., 2018)"""

    def __init__(
        self,
        *,
        vocab: Dict[str, int],
        special_pieces: Optional[Dict[str, int]],
    ):
        """Construct a tokenizer from a curated tokenizers WordPiece processor.

        :param vocab:
            The word piece vocabulary.
        :param special_pieces:
            Additional pieces.
        """
        # Added tokens are usually already in the wordpiece vocabs, but lets
        # just add them to be sure.
        self.special_piece_to_id = {} if special_pieces is None else special_pieces
        self.id_to_special_piece = {v: k for k, v in self.special_piece_to_id.items()}
        vocab.update(self.special_piece_to_id)

        # We'll build up the vocab, verifying that the user provided ids for
        # all tokens as a sanity check.
        vocab_size = max(vocab.values()) + 1
        pieces: List[Optional[str]] = [None] * vocab_size
        for piece, idx in vocab.items():
            pieces[idx] = piece

        unused_indices = [str(id) for id, piece in enumerate(pieces) if piece is None]
        if unused_indices:
            raise ValueError(
                f"WordPiece vocabulary contains unused indices: {', '.join(unused_indices)}"
            )

        self.processor = WordPieceProcessor(pieces)

    def _decode(
        self, input: Iterable[Iterable[int]], skip_special_pieces: bool
    ) -> List[str]:
        decoded = []
        for piece_ids in input:
            tokens = []
            for piece_id in piece_ids:
                if piece_id in self.id_to_special_piece:
                    continue

                token, is_initial = self.processor.id_to_piece(piece_id)
                if is_initial:
                    token = " " + token
                tokens.append(token)
            decoded.append("".join(tokens))
        return decoded

    def _encode(self, input: Iterable[MergedInputChunks]) -> PiecesWithIds:
        ids = []
        pieces = []

        for seq in input:
            seq_ids = []
            seq_pieces = []

            for chunk in seq:
                if isinstance(chunk, MergedSpecialPieceChunk):
                    seq_ids.append(self.processor.get_initial(chunk.piece))
                    seq_pieces.append(chunk.piece)
                else:
                    for token in chunk.text.split(" "):
                        token_ids, token_pieces = self.processor.encode(token)
                        seq_ids.extend(token_ids)
                        seq_pieces.extend(token_pieces)

            ids.append(seq_ids)
            pieces.append(seq_pieces)

        return PiecesWithIds(ids=ids, pieces=pieces)


def clean_up_decoded_string_like_hf(text: str) -> str:
    """This method is provided to ensure that the decoded string
    is compatible with the decoded output of the corresponding HF
    WordPiece tokenizer.

    c.f https://github.com/huggingface/tokenizers/blob/b4fcc9ce6e4ad5806e82826f816acfdfdc4fcc67/tokenizers/src/decoders/wordpiece.rs#L31
    """
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
