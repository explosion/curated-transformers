from typing import Callable, Iterable, List

from .tokenizer import PiecesWithIds, PostEncoder, PreDecoder
from .util import remove_pieces_from_sequence, add_bos_eos_to_encoding


class FAIRSEQ_PIECE_IDS:
    FAIRSEQ_BOS = 0
    FAIRSEQ_EOS = 2
    FAIRSEQ_UNK = 3

    SPP_BOS = 1
    SPP_EOS = 2
    SPP_UNK = 0


class FairSeqPostEncoder(PostEncoder):
    """Performs fixups of SentencePiece piece identifiers for models that use
    the fairseq vocabulary."""

    def __init__(
        self,
        *,
        bos_piece: str,
        eos_piece: str,
        bos_id: int,
        eos_id: int,
        piece_updater: Callable[[int], int],
    ):
        """Construct a fairseq post-encoder.

        bos_piece (str): The piece used to mark the beginning of a sequence.
        eos_piece (str): The piece used to mark the end of a sequence.
        bos_id (int): The piece id used to mark the beginning of a sequence.
        eos_id (int): The piece id used to mark the end of a sequence.
        piece_updater (Callable[[int], int]): Function that tranforms a given
            SentencePiece piece identifier to a valid fairseq one.
        """
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.piece_updater = piece_updater

    def __call__(self, pieces_with_ids: PiecesWithIds) -> PiecesWithIds:
        pieces_with_ids = add_bos_eos_to_encoding(
            pieces_with_ids,
            bos_piece=self.bos_piece,
            eos_piece=self.eos_piece,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )

        # We need to align the IDs to the original fairseq vocabulary.
        for piece_ids in pieces_with_ids.ids:
            for i in range(len(piece_ids)):
                piece_ids[i] = self.piece_updater(piece_ids[i])

        return pieces_with_ids


class FairSeqPreDecoder(PreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
        piece_updater: Callable[[int], int],
    ):
        """Construct a fairseq pre-decoder.

        bos_id (int): The piece id used to mark the beginning of a sequence.
        eos_id (int): The piece id used to mark the end of a sequence.
        piece_updater (Callable[[int], int]): Function that tranforms a given
            fairseq piece identifier to the original SentencePiece one.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.piece_updater = piece_updater

    def __call__(self, input: Iterable[Iterable[int]]) -> List[List[int]]:
        # Revert the fairseq alignment.
        input = (
            (self.piece_updater(piece_id) for piece_id in piece_ids)
            for piece_ids in input
        )

        return [
            list(remove_pieces_from_sequence(ids, (self.bos_id, self.eos_id)))
            for ids in input
        ]
