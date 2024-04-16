from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from curated_tokenizers import SentencePieceProcessor

from ...repository.file import RepositoryFile
from ..hf_hub import LegacyFromHF
from ._fairseq import FAIRSEQ_PIECE_IDS, FairSeqPostEncoder, FairSeqPreDecoder
from .legacy_tokenizer import AddBosEosPreEncoder
from .sentencepiece_tokenizer import SentencePieceTokenizer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="CamemBERTTokenizer")


_CAMEMBERT_FAIRSEQ_OFFSET = 4


class CamemBERTPostEncoder(FairSeqPostEncoder):
    def __init__(
        self,
    ):
        """
        Construct a CamemBERT post-encoder.
        """
        super(CamemBERTPostEncoder, self).__init__(
            piece_updater=CamemBERTPostEncoder._sentencepiece_to_fairseq,
        )

    @staticmethod
    def _sentencepiece_to_fairseq(piece_id: int):
        if piece_id == FAIRSEQ_PIECE_IDS.SPP_UNK:
            return FAIRSEQ_PIECE_IDS.FAIRSEQ_UNK
        else:
            return piece_id + _CAMEMBERT_FAIRSEQ_OFFSET


class CamemBERTPreDecoder(FairSeqPreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
    ):
        """
        Construct a CamemBERT pre-decoder.

        :param bos_id:
            The piece id used to mark the beginning of a sequence.
        :param eos_id:
            The piece id used to mark the end of a sequence.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id
        super(CamemBERTPreDecoder, self).__init__(
            bos_id=bos_id,
            eos_id=eos_id,
            piece_updater=CamemBERTPreDecoder._fairseq_to_sentencepiece,
        )

    @staticmethod
    def _fairseq_to_sentencepiece(piece_id: int):
        if piece_id == FAIRSEQ_PIECE_IDS.FAIRSEQ_UNK:
            return FAIRSEQ_PIECE_IDS.SPP_UNK
        else:
            return piece_id - _CAMEMBERT_FAIRSEQ_OFFSET


class CamemBERTTokenizer(SentencePieceTokenizer, LegacyFromHF):
    """
    Legacy tokenizer for CamemBERT (`Martin et al., 2020`_) models.

    .. _Martin et al., 2020: https://arxiv.org/abs/1911.03894
    """

    vocab_files: Dict[str, str] = {"model": "sentencepiece.bpe.model"}

    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ):
        """
        Construct a CamemBERT tokenizer from a ``curated-tokenizers`` SentencePiece processor.

        :param processor:
            The processor to wrap.
        :param bos_piece:
            The piece to use to mark the beginning of a sequence.
        :param eos_piece:
            The piece to use to mark the end of a sequence.
        """
        super().__init__(processor=processor)

        self.processor = processor

        bos_id = _get_piece_id_or_fail(processor, bos_piece)
        eos_id = _get_piece_id_or_fail(processor, eos_piece)

        self.pre_encoder = AddBosEosPreEncoder(bos_piece=bos_piece, eos_piece=eos_piece)

        self.post_encoder = CamemBERTPostEncoder()

        self.pre_decoder = CamemBERTPreDecoder(
            bos_id=bos_id,
            eos_id=eos_id,
        )

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        model_file: RepositoryFile,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ) -> Self:
        """
        Construct a tokenizer from vocabulary and merge files.

        :param model_file:
            The SentencePiece model file.
        :param bos_piece:
            The piece to use to mark the beginning of a sequence.
        :param eos_piece:
            The piece to use to mark the end of a sequence.
        """
        with model_file.open() as f:
            processor = SentencePieceProcessor.from_file(f)
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
        )

    @classmethod
    def _load_from_vocab_files(
        cls: Type[Self],
        *,
        vocab_files: Mapping[str, RepositoryFile],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> Self:
        return cls.from_files(model_file=vocab_files["model"])


def _get_piece_id_or_fail(processor: SentencePieceProcessor, piece: str) -> int:
    piece_id = processor.piece_to_id(piece)
    if piece_id == processor.unk_id():
        raise ValueError(
            f"CamemBERT piece encoder vocabulary doesn't contain '{piece}' piece"
        )
    return piece_id
