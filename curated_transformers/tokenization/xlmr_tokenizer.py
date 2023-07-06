from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from curated_tokenizers import SentencePieceProcessor

from ._fairseq import FAIRSEQ_PIECE_IDS, FairSeqPostEncoder, FairSeqPreDecoder
from .hf_hub import LegacyFromHFHub
from .sentencepiece_tokenizer import SentencePieceTokenizer
from .tokenizer import AddBosEosPreEncoder

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="XlmrTokenizer")


_XLMR_FAIRSEQ_OFFSET = 1


class XlmrPostEncoder(FairSeqPostEncoder):
    def __init__(
        self,
    ):
        """
        Construct a XLM-R post-encoder.
        """
        super(XlmrPostEncoder, self).__init__(
            piece_updater=XlmrPostEncoder._sentencepiece_to_fairseq,
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


class XlmrPreDecoder(FairSeqPreDecoder):
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
        super(XlmrPreDecoder, self).__init__(
            bos_id=bos_id,
            eos_id=eos_id,
            piece_updater=XlmrPreDecoder._fairseq_to_sentencepiece,
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


class XlmrTokenizer(SentencePieceTokenizer, LegacyFromHFHub):
    """
    Legacy tokenizer for XLM-RoBERTa (Conneau et al., 2019).
    """

    vocab_files: Dict[str, str] = {"model": "sentencepiece.bpe.model"}

    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
    ):
        """
        Construct a XLM-R tokenizer.

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
        self.post_encoder = XlmrPostEncoder()
        self.pre_decoder = XlmrPreDecoder(bos_id=bos_id, eos_id=eos_id)

        self._eos_piece = eos_piece

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        model_path: Path,
    ) -> Self:
        """
        Construct a XLM-R tokenizer from a SentencePiece model.

        :param model_path:
            Path to the SentencePiece model file.
        """
        processor = SentencePieceProcessor.from_file(str(model_path))
        return cls(processor=processor)

    @classmethod
    def _load_from_vocab_files(
        cls: Type[Self],
        *,
        vocab_files: Dict[str, Path],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> Self:
        return cls.from_files(model_path=vocab_files["model"])
