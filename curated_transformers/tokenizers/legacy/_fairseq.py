from typing import Callable, Iterable, List

from ..tokenizer import PiecesWithIds
from ..util import remove_pieces_from_sequence
from .legacy_tokenizer import PostEncoder, PreDecoder


class FAIRSEQ_PIECE_IDS:
    FAIRSEQ_BOS = 0
    FAIRSEQ_EOS = 2
    FAIRSEQ_UNK = 3

    SPP_BOS = 1
    SPP_EOS = 2
    SPP_UNK = 0


class FairSeqPostEncoder(PostEncoder):
    """
    Performs fixups of SentencePiece piece identifiers for models that use
    the fairseq vocabulary.
    """

    def __init__(
        self,
        *,
        piece_updater: Callable[[int], int],
    ):
        """
        Construct a fairseq post-encoder.

        :param piece_updater:
            Function that tranforms a given SentencePiece piece
            identifier to a valid fairseq one.
        """
        self.piece_updater = piece_updater

    def __call__(self, pieces_with_ids: PiecesWithIds) -> PiecesWithIds:
        # We need to align the IDs to the original fairseq vocabulary.
        for piece_ids in pieces_with_ids.ids:
            for i in range(len(piece_ids)):
                piece_ids[i] = self.piece_updater(piece_ids[i])

        return pieces_with_ids


class FairSeqPreDecoder(PreDecoder):
    """
    Performs fixups of SentencePiece piece identifiers for models that use
    the fairseq vocabulary.
    """

    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
        piece_updater: Callable[[int], int],
    ):
        """
        Construct a fairseq pre-decoder.

        :param bos_id:
            The piece id used to mark the beginning of a sequence.
        :param eos_id:
            The piece id used to mark the end of a sequence.
        :param piece_updater:
            Function that tranforms a given fairseq piece identifier
            to the original SentencePiece one.
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
