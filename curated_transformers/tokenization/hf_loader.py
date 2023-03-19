from typing import Callable, Optional
import json


from .._compat import has_hf_transformers, transformers
from .bbpe_encoder import ByteBPEProcessor
from .sentencepiece_encoder import SentencePieceProcessor
from .wordpiece_encoder import WordPieceProcessor
from .types import (
    Tok2PiecesInT,
    Tok2PiecesModelT,
)
from ..errors import Errors

if has_hf_transformers:
    SUPPORTED_TOKENIZERS = (
        transformers.BertTokenizerFast,
        transformers.RobertaTokenizerFast,
        transformers.XLMRobertaTokenizerFast,
        transformers.CamembertTokenizerFast,
        transformers.BertJapaneseTokenizer,
    )
else:
    SUPPORTED_TOKENIZERS = ()  # type: ignore


def build_hf_piece_encoder_loader_v1(
    *, name: str, revision: str = "main"
) -> Callable[
    [Tok2PiecesModelT, Optional[Tok2PiecesInT], Optional[Tok2PiecesInT]],
    Tok2PiecesModelT,
]:
    """Construct a callback that initializes a HuggingFace piece encoder
    model. Used in conjunction with the HuggingFace model loader.

    name (str):
        Name of the HuggingFace model.
    revision (str):
        Name of the model revision/branch.
    """

    def load(model, X=None, Y=None):
        if not has_hf_transformers:
            raise ValueError(Errors.E011.format(loader_name="HFPieceEncoderLoader"))

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
        return _convert_sentencepiece_encoder(model, tokenizer)  # type: ignore
    elif isinstance(tokenizer, transformers.BertJapaneseTokenizer):
        return _convert_bert_japanese_encoder(model, tokenizer)
    else:
        raise ValueError(
            Errors.E022.format(
                unsupported_tokenizer=type(tokenizer),
                supported_tokenizers=SUPPORTED_TOKENIZERS,
            )
        )


def _convert_byte_bpe_encoder(
    model: Tok2PiecesModelT,
    tokenizer: "transformers.RobertaTokenizerFast",
) -> Tok2PiecesModelT:
    # Seems like we cannot get the vocab file name for a RoBERTa vocabulary? And
    # neither the merges from the fast tokenizer. So we'll get them from the
    # JSON serialization.
    serialized = tokenizer.backend_tokenizer.to_str(True)  # type: ignore
    deserialized = json.loads(serialized)
    vocab_merges = deserialized["model"]
    merges = [tuple(merge.split(" ")) for merge in vocab_merges["merges"]]
    model.attrs["byte_bpe_processor"] = ByteBPEProcessor(vocab_merges["vocab"], merges)
    model.attrs["bos_piece"] = tokenizer.bos_token  # type: ignore
    model.attrs["eos_piece"] = tokenizer.eos_token  # type: ignore
    model.attrs["unk_piece"] = tokenizer.unk_token  # type: ignore

    return model


def _convert_sentencepiece_encoder(
    model: Tok2PiecesModelT,
    tokenizer: "transformers.RobertaTokenizerFast",
) -> Tok2PiecesModelT:
    model.get_ref("encoder").attrs[
        "sentencepiece_processor"
    ] = SentencePieceProcessor.from_file(
        tokenizer.vocab_file  # type: ignore
    )
    return model


def _convert_wordpiece_encoder(
    model: Tok2PiecesModelT, tokenizer: "transformers.BertTokenizerFast"
) -> Tok2PiecesModelT:
    # Seems like we cannot get the vocab file name for a BERT vocabulary? So,
    # instead, copy the vocabulary.
    vocab = [None] * tokenizer.vocab_size  # type: ignore
    for piece, idx in tokenizer.vocab.items():  # type: ignore
        vocab[idx] = piece

    strip_accents = tokenizer.backend_tokenizer.normalizer.strip_accents  # type: ignore
    lowercase = tokenizer.do_lower_case  # type: ignore
    model.attrs["wordpiece_processor"] = WordPieceProcessor(vocab)
    model.attrs["bos_piece"] = tokenizer.cls_token  # type: ignore
    model.attrs["eos_piece"] = tokenizer.sep_token  # type: ignore
    model.attrs["unk_piece"] = tokenizer.unk_token  # type: ignore
    model.attrs["lowercase"] = lowercase

    # Huggingface BERT also strips accents when lowercasing is enabled
    # and accent stripping is not defined.
    model.attrs["strip_accents"] = strip_accents or (
        strip_accents is not False and lowercase
    )

    return model


def _convert_bert_japanese_encoder(
    model: Tok2PiecesModelT, tokenizer: "transformers.BertJapaneseTokenizer"
) -> Tok2PiecesModelT:
    if not isinstance(
        tokenizer.subword_tokenizer,
        transformers.models.bert_japanese.CharacterTokenizer,
    ):
        raise ValueError(Errors.E023)
    if model.name != "char_encoder":
        raise ValueError(Errors.E024.format(model_name=model.name))

    model.attrs["bos_piece"] = tokenizer.cls_token
    model.attrs["eos_piece"] = tokenizer.sep_token
    model.attrs["unk_piece"] = tokenizer.unk_token
    model.attrs["normalize"] = (
        "NFKC" if tokenizer.subword_tokenizer.normalize_text else None
    )
    model.attrs["vocab"] = tokenizer.vocab.copy()

    return model
