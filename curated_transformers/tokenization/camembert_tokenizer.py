from typing import Any, Type, TypeVar
from cutlery import SentencePieceProcessor
from pathlib import Path

from ._fairseq import FairSeqPostEncoder, FairSeqPreDecoder, FAIRSEQ_PIECE_IDS
from .sentencepiece_tokenizer import SentencePieceTokenizer
from .hf_hub import FromPretrainedHFTokenizer


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="CamembertTokenizer")


_CAMEMBERT_FAIRSEQ_OFFSET = 4


class CamembertPostEncoder(FairSeqPostEncoder):
    def __init__(
        self,
        *,
        bos_piece: str,
        eos_piece: str,
        bos_id: int,
        eos_id: int,
    ):
        """Construct a CamemBERT post-encoder.

        bos_piece (str): The piece used to mark the beginning of a sequence.
        eos_piece (str): The piece used to mark the end of a sequence.
        bos_id (int): The piece id used to mark the beginning of a sequence.
        eos_id (int): The piece id used to mark the end of a sequence.
        """
        super(CamembertPostEncoder, self).__init__(
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            bos_id=bos_id,
            eos_id=eos_id,
            piece_updater=CamembertPostEncoder._sentencepiece_to_fairseq,
        )

    @staticmethod
    def _sentencepiece_to_fairseq(piece_id: int):
        if piece_id == FAIRSEQ_PIECE_IDS.SPP_UNK:
            return FAIRSEQ_PIECE_IDS.FAIRSEQ_UNK
        else:
            return piece_id + _CAMEMBERT_FAIRSEQ_OFFSET


class CamembertPreDecoder(FairSeqPreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
    ):
        """Construct a CamemBERT pre-decoder.

        bos_id (int): The piece id used to mark the beginning of a sequence.
        eos_id (int): The piece id used to mark the end of a sequence.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id
        super(CamembertPreDecoder, self).__init__(
            bos_id=bos_id,
            eos_id=eos_id,
            piece_updater=CamembertPreDecoder._fairseq_to_sentencepiece,
        )

    @staticmethod
    def _fairseq_to_sentencepiece(piece_id: int):
        if piece_id == FAIRSEQ_PIECE_IDS.FAIRSEQ_UNK:
            return FAIRSEQ_PIECE_IDS.SPP_UNK
        else:
            return piece_id - _CAMEMBERT_FAIRSEQ_OFFSET


class CamembertTokenizer(SentencePieceTokenizer, FromPretrainedHFTokenizer):
    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ):
        """Construct a CamemBERT tokenizer from a cutlery SentencePiece processor.

        processor (SentencePieceProcessor): The processor to wrap.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        """
        super().__init__(processor=processor)

        self.processor = processor

        bos_id = _get_piece_id_or_fail(processor, bos_piece)
        eos_id = _get_piece_id_or_fail(processor, eos_piece)

        self.post_encoder = CamembertPostEncoder(
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            bos_id=bos_id,
            eos_id=eos_id,
        )

        self.pre_decoder = CamembertPreDecoder(
            bos_id=bos_id,
            eos_id=eos_id,
        )

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        model_path: Path,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ) -> Self:
        """Construct a tokenizer from vocabulary and merge files.

        model_path (Path): Path to the SentencePiece model file.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        """
        processor = SentencePieceProcessor.from_file(str(model_path))
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
        )

    @classmethod
    def convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        if not hasattr(tokenizer, "vocab_file"):
            raise ValueError(
                f"Hugging Face tokenizer (`{type(tokenizer)}`) doesn't "
                "contain the path to the SentencePiece model file"
            )
        model_path = tokenizer.vocab_file
        processor = SentencePieceProcessor.from_file(model_path)
        bos_piece = tokenizer.bos_token  # type: ignore
        eos_piece = tokenizer.eos_token  # type: ignore
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
        )


def _get_piece_id_or_fail(processor: SentencePieceProcessor, piece: str) -> int:
    piece_id = processor.piece_to_id(piece)
    if piece_id == processor.unk_id():
        raise ValueError(
            f"CamemBERT piece encoder vocabulary doesn't contain '{piece}' piece"
        )
    return piece_id
