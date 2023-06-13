from pathlib import Path
from typing import Any, Type, TypeVar

from curated_tokenizers import SentencePieceProcessor

from ._fairseq import FAIRSEQ_PIECE_IDS, FairSeqPostEncoder, FairSeqPreDecoder
from .hf_hub import FromPretrainedHFTokenizer
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
        """Construct a XLM-R pre-decoder.

        :param bos_id: The piece id used to mark the beginning of a sequence.
        :param eos_id: The piece id used to mark the end of a sequence.
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


class XlmrTokenizer(SentencePieceTokenizer, FromPretrainedHFTokenizer):
    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
    ):
        """Construct a XLM-R tokenizer from a curated tokenizers SentencePiece processor.

        :param processor: The processor to wrap.
        :param bos_piece: The piece to use to mark the beginning of a sequence.
        :param eos_piece: The piece to use to mark the end of a sequence.
        """
        super().__init__(processor=processor)

        self.processor = processor

        bos_id = _get_piece_id_or_fail(processor, bos_piece)
        eos_id = _get_piece_id_or_fail(processor, eos_piece)

        self.pre_encoder = AddBosEosPreEncoder(bos_piece=bos_piece, eos_piece=eos_piece)

        self.post_encoder = XlmrPostEncoder()

        self.pre_decoder = XlmrPreDecoder(
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

        :param model_path: Path to the SentencePiece model file.
        :param bos_piece: The piece to use to mark the beginning of a sequence.
        :param eos_piece: The piece to use to mark the end of a sequence.
        """
        processor = SentencePieceProcessor.from_file(str(model_path))
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
        )

    @classmethod
    def _convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
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
            f"XLM-R piece encoder vocabulary doesn't contain '{piece}' piece"
        )
    return piece_id
