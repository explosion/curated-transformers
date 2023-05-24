from typing import Iterable, Tuple

from .tokenizer import PiecesWithIds


def remove_pieces_from_sequence(
    input: Iterable[int], pieces_to_remove: Tuple[int, ...]
) -> Iterable[int]:
    return (piece for piece in input if piece not in pieces_to_remove)
