from typing import Any, Callable, List, Optional
import json
from spacy.tokens import Doc
from thinc.api import Model
from thinc.types import Ragged

from .._compat import has_hf_transformers, transformers
from ..util import registry
from .bbpe_encoder import ByteBPEProcessor
from .sentencepiece_encoder import SentencePieceProcessor
from .wordpiece_encoder import WordPieceProcessor


def build_hf_piece_encoder_loader_v1(
    *, name: str, revision: str = "main"
) -> Callable[
    [Model[List[Doc], List[Ragged]], Optional[List[Doc]], Any],
    Model[List[Doc], List[Ragged]],
]:
    def load(
        model: Model[List[Doc], List[Ragged]], X: List[Doc] = None, Y=None
    ) -> Model[List[Doc], List[Ragged]]:
        if not has_hf_transformers:
            raise ValueError("requires transformers package")

        tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        return _convert_encoder(model, tokenizer)

    return load


def _convert_encoder(
    model: Model, tokenizer: "transformers.PreTrainedTokenizerBase"
) -> Model[List[Doc], List[Ragged]]:
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
    model: Model[List[Doc], List[Ragged]],
    tokenizer: "transformers.RobertaTokenizerFast",
) -> Model[List[Doc], List[Ragged]]:
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
    model: Model[List[Doc], List[Ragged]],
    tokenizer: "transformers.RobertaTokenizerFast",
) -> Model[List[Doc], List[Ragged]]:
    model.attrs["sentencepiece_processor"] = SentencePieceProcessor.from_file(
        tokenizer.vocab_file
    )
    return model


def _convert_wordpiece_encoder(
    model: Model[List[Doc], List[Ragged]], tokenizer: "transformers.BertTokenizerFast"
) -> Model[List[Doc], List[Ragged]]:
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
