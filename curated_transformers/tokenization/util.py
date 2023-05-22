from typing import Iterable, Tuple

from .tokenizer import PiecesWithIds


def remove_pieces_from_sequence(
    input: Iterable[int], pieces_to_remove: Tuple[int, ...]
) -> Iterable[int]:
    return (piece for piece in input if piece not in pieces_to_remove)


def add_bos_eos_to_encoding(
    pieces_with_ids: PiecesWithIds,
    bos_piece: str,
    eos_piece: str,
    bos_id: int,
    eos_id: int,
) -> PiecesWithIds:
    ids = []
    for seq_ids in pieces_with_ids.ids:
        ids.append([bos_id] + seq_ids + [eos_id])
    pieces = []
    for seq_pieces in pieces_with_ids.pieces:
        pieces.append([bos_piece] + seq_pieces + [eos_piece])
    lens = []
    for seq_lens in pieces_with_ids.lens:
        lens.append([1] + seq_lens + [1])

    return PiecesWithIds(ids=ids, lens=lens, pieces=pieces)
