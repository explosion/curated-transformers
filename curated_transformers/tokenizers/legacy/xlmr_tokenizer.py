from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from curated_tokenizers import SentencePieceProcessor

from ...repository.file import RepositoryFile
from ..hf_hub import LegacyFromHF
from ._fairseq import FAIRSEQ_PIECE_IDS, FairSeqPostEncoder, FairSeqPreDecoder
from .legacy_tokenizer import AddBosEosPreEncoder
from .sentencepiece_tokenizer import SentencePieceTokenizer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="XLMRTokenizer")


_XLMR_FAIRSEQ_OFFSET = 1


class XLMRPostEncoder(FairSeqPostEncoder):
    def __init__(
        self,
    ):
        """
        Construct a XLM-R post-encoder.
        """
        super(XLMRPostEncoder, self).__init__(
            piece_updater=XLMRPostEncoder._sentencepiece_to_fairseq,
        )

    @staticmethod
    def _sentencepiece_to_fairseq(piece_id: int):
        if piece_id == FAIRSEQ_PIECE_IDS.SPP_UNK:
            return FAIRSEQ_PIECE_IDS.FAIRSEQ_UNK
        elif piece_id == FAIRSEQ_PIECE_IDS.SPP_BOS:
            return FAIRSEQ_PIECE_IDS.FAIRSEQ_BOS
        elif piece_id == FAIRSEQ_PIECE_IDS.SPP_EOS:
            return FAIRSEQ_PIECE_IDS.FAIRSEQ_EOS
        else:
            return piece_id + _XLMR_FAIRSEQ_OFFSET


class XLMRPreDecoder(FairSeqPreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
    ):
        """
        Construct a XLM-R pre-decoder.

        :param bos_id:
            The piece id used to mark the beginning of a sequence.
        :param eos_id:
            The piece id used to mark the end of a sequence.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id
        super(XLMRPreDecoder, self).__init__(
            bos_id=bos_id,
            eos_id=eos_id,
            piece_updater=XLMRPreDecoder._fairseq_to_sentencepiece,
        )

    @staticmethod
    def _fairseq_to_sentencepiece(piece_id: int):
        if piece_id == FAIRSEQ_PIECE_IDS.FAIRSEQ_UNK:
            return FAIRSEQ_PIECE_IDS.SPP_UNK
        elif piece_id == FAIRSEQ_PIECE_IDS.FAIRSEQ_BOS:
            return FAIRSEQ_PIECE_IDS.SPP_BOS
        elif piece_id == FAIRSEQ_PIECE_IDS.FAIRSEQ_EOS:
            return FAIRSEQ_PIECE_IDS.SPP_EOS
        else:
            return piece_id - _XLMR_FAIRSEQ_OFFSET


class XLMRTokenizer(SentencePieceTokenizer, LegacyFromHF):
    """
    Legacy tokenizer for XLM-RoBERTa (`Conneau et al., 2019`_) models.

    .. _Conneau et al., 2019: https://arxiv.org/abs/1911.02116
    """

    vocab_files: Dict[str, str] = {"model": "sentencepiece.bpe.model"}

    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
    ):
        """
        Construct a XLM-RoBERTa tokenizer from a ``curated-tokenizers`` SentencePiece processor.

        :param processor:
            The processor to wrap.
        """
        super().__init__(processor=processor)

        self.processor = processor

        bos_id = processor.bos_id()
        eos_id = processor.eos_id()
        bos_piece = processor.id_to_piece(bos_id)
        eos_piece = processor.id_to_piece(eos_id)

        self.pre_encoder = AddBosEosPreEncoder(bos_piece=bos_piece, eos_piece=eos_piece)
        self.post_encoder = XLMRPostEncoder()
        self.pre_decoder = XLMRPreDecoder(bos_id=bos_id, eos_id=eos_id)

        self._eos_piece = eos_piece

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        model_file: RepositoryFile,
    ) -> Self:
        """
        Construct a XLM-R tokenizer from a SentencePiece model.

        :param model_file:
            The SentencePiece model file.
        """
        with model_file.open() as f:
            processor = SentencePieceProcessor.from_file(f)
        return cls(processor=processor)

    @classmethod
    def _load_from_vocab_files(
        cls: Type[Self],
        *,
        vocab_files: Mapping[str, RepositoryFile],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> Self:
        return cls.from_files(model_file=vocab_files["model"])
