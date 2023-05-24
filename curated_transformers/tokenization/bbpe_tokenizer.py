from typing import Iterable, List
from curated_tokenizers import ByteBPEProcessor

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

    def _encode(self, input: Iterable[str]) -> PiecesWithIds:
        ids = []
        pieces = []

        for text in input:
            text_ids = []
            text_pieces = []

            for idx, token in enumerate(text.split(" ")):
                if idx != 0:
                    token = " " + token
                token_ids, token_pieces = self.processor.encode(token)
                text_ids.extend(token_ids)
                text_pieces.extend(token_pieces)

            ids.append(text_ids)
            pieces.append(text_pieces)

        return PiecesWithIds(ids=ids, pieces=pieces)
