from typing import Callable, Optional
import json

from .._compat import has_hf_transformers, transformers
from ..util import registry
from .bbpe_encoder import ByteBPEProcessor
from .sentencepiece_encoder import SentencePieceProcessor
from .wordpiece_encoder import WordPieceProcessor
from .types import (
    Tok2PiecesInT,
    Tok2PiecesModelT,
)


def build_hf_piece_encoder_loader_v1(
    *, name: str, revision: str = "main"
) -> Callable[
    [Tok2PiecesModelT, Optional[Tok2PiecesInT], Optional[Tok2PiecesInT]],
    Tok2PiecesModelT,
]:
    def load(model, X=None, Y=None):
        if not has_hf_transformers:
            raise ValueError("requires transformers package")

        tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        return _convert_encoder(model, tokenizer)

    return load


def _convert_encoder(
    model: Tok2PiecesModelT, tokenizer: "transformers.PreTrainedTokenizerBase"
) -> Tok2PiecesModelT:
    if isinstance(tokenizer, transformers.BertTokenizerFast):
        return _convert_wordpiece_encoder(model, tokenizer)
    elif isinstance(tokenizer, transformers.RobertaTokenizerFast):
        return _convert_byte_bpe_encoder(model, tokenizer)
    elif isinstance(
        tokenizer,
        (transformers.XLMRobertaTokenizerFast, transformers.CamembertTokenizerFast),
    ):
        return _convert_sentencepiece_encoder(model, tokenizer)

    raise ValueError(
        f"Loading from the '{type(tokenizer)}' huggingface tokenizer is not supported"
    )


def _convert_byte_bpe_encoder(
    model: Tok2PiecesModelT,
    tokenizer: "transformers.RobertaTokenizerFast",
) -> Tok2PiecesModelT:
    # Seems like we cannot get the vocab file name for a RoBERTa vocabulary? And
    # neither the merges from the fast tokenizer. So we'll get them from the
    # JSON serialization.
    serialized = tokenizer.backend_tokenizer.to_str(True)
    deserialized = json.loads(serialized)
    vocab_merges = deserialized["model"]
    merges = [tuple(merge.split(" ")) for merge in vocab_merges["merges"]]
    model.attrs["byte_bpe_processor"] = ByteBPEProcessor(vocab_merges["vocab"], merges)
    model.attrs["bos_piece"] = tokenizer.bos_token
    model.attrs["eos_piece"] = tokenizer.eos_token
    model.attrs["unk_piece"] = tokenizer.unk_token

    return model


def _convert_sentencepiece_encoder(
    model: Tok2PiecesModelT,
    tokenizer: "transformers.RobertaTokenizerFast",
) -> Tok2PiecesModelT:
    model.attrs["sentencepiece_processor"] = SentencePieceProcessor.from_file(
        tokenizer.vocab_file
    )
    return model


def _convert_wordpiece_encoder(
    model: Tok2PiecesModelT, tokenizer: "transformers.BertTokenizerFast"
) -> Tok2PiecesModelT:
    # Seems like we cannot get the vocab file name for a BERT vocabulary? So,
    # instead, copy the vocabulary.
    vocab = [None] * tokenizer.vocab_size
    for piece, idx in tokenizer.vocab.items():
        vocab[idx] = piece
    model.attrs["wordpiece_processor"] = WordPieceProcessor(vocab)
    model.attrs["bos_piece"] = tokenizer.cls_token
    model.attrs["eos_piece"] = tokenizer.sep_token
    model.attrs["unk_piece"] = tokenizer.unk_token

    return model
