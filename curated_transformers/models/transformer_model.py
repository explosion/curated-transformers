from typing import Callable, List, Optional, Tuple
from functools import partial
import torch

from spacy.tokens import Span, Doc
from thinc.layers import chain
from thinc.model import Model
from thinc.types import Ragged

from ..models.albert import AlbertConfig, AlbertEncoder
from ..models.bert import BertConfig, BertEncoder
from ..models.roberta import RobertaConfig, RobertaEncoder, QuantizedRobertaEncoder
from ..models.torchscript_wrapper import TorchScriptWrapper_v1
from ..tokenization.sentencepiece_adapters import build_xlmr_adapter, remove_bos_eos
from ..tokenization.sentencepiece_encoder import build_sentencepiece_encoder
from ..tokenization.wordpiece_encoder import build_wordpiece_encoder
from .hf_wrapper import (
    _convert_inputs,
    _convert_outputs,
    build_hf_transformer_encoder_v1,
)


def build_albert_transformer_model_v1(
    *,
    vocab_size,
    with_spans,
    attention_probs_dropout_prob: float = 0.0,
    embedding_size=128,
    hidden_act: str = "gelu_new",
    hidden_dropout_prob: float = 0.0,
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    layer_norm_eps: float = 1e-12,
    max_position_embeddings: int = 512,
    model_max_length: int = 512,
    num_attention_heads: int = 12,
    num_hidden_groups: int = 1,
    num_hidden_layers: int = 12,
    padding_idx: int = 0,
    type_vocab_size: int = 2,
    torchscript=False,
):
    config = AlbertConfig(
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_groups=num_hidden_groups,
        num_hidden_layers=num_hidden_layers,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        hidden_act=hidden_act,
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        model_max_length=model_max_length,
        layer_norm_eps=layer_norm_eps,
        padding_idx=padding_idx,
    )

    if torchscript:
        transformer = _torchscript_encoder(
            model_max_length=model_max_length, padding_idx=padding_idx
        )
    else:
        encoder = AlbertEncoder(config)
        transformer = build_hf_transformer_encoder_v1(encoder)

    piece_encoder = build_sentencepiece_encoder()

    return build_transformer_model_v1(
        with_spans=with_spans,
        piece_encoder=piece_encoder,
        transformer=transformer,
    )


def build_bert_transformer_model_v1(
    *,
    vocab_size,
    with_spans,
    attention_probs_dropout_prob: float = 0.1,
    hidden_act: str = "gelu",
    hidden_dropout_prob: float = 0.1,
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    layer_norm_eps: float = 1e-12,
    max_position_embeddings: int = 512,
    model_max_length: int = 512,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    padding_idx: int = 0,
    type_vocab_size: int = 2,
    torchscript=False,
):
    config = BertConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        hidden_act=hidden_act,
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        model_max_length=model_max_length,
        layer_norm_eps=layer_norm_eps,
        padding_idx=padding_idx,
    )

    if torchscript:
        transformer = _torchscript_encoder(
            model_max_length=model_max_length, padding_idx=padding_idx
        )
    else:
        encoder = BertEncoder(config)
        transformer = build_hf_transformer_encoder_v1(encoder)

    piece_encoder = build_wordpiece_encoder()

    return build_transformer_model_v1(
        with_spans=with_spans,
        piece_encoder=piece_encoder,
        transformer=transformer,
    )


def build_xlmr_transformer_model_v1(
    *,
    vocab_size,
    with_spans,
    attention_probs_dropout_prob: float = 0.1,
    hidden_act: str = "gelu",
    hidden_dropout_prob: float = 0.1,
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    layer_norm_eps: float = 1e-5,
    max_position_embeddings: int = 514,
    model_max_length: int = 512,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    padding_idx: int = 1,
    type_vocab_size: int = 1,
    torchscript=False,
    qat=False,
):
    piece_adapter = build_xlmr_adapter()

    config = RobertaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        hidden_act=hidden_act,
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        model_max_length=model_max_length,
        layer_norm_eps=layer_norm_eps,
        padding_idx=padding_idx,
    )

    if torchscript:
        transformer = _torchscript_encoder(
            model_max_length=model_max_length, padding_idx=padding_idx
        )
    else:
        if qat:
            encoder = QuantizedRobertaEncoder(config)
            #encoder = torch.quantization.prepare_qat(encoder)
        else:
            encoder = RobertaEncoder(config)
        transformer = build_hf_transformer_encoder_v1(encoder)

    piece_encoder = build_sentencepiece_encoder()

    return build_transformer_model_v1(
        with_spans=with_spans,
        piece_encoder=piece_encoder,
        piece_adapter=piece_adapter,
        transformer=transformer,
    )


def build_transformer_model_v1(
    *,
    with_spans,
    piece_encoder: Model[List[Span], List[Ragged]],
    piece_adapter: Model[List[Ragged], List[Ragged]] = None,
    transformer: Model[List[Ragged], List[Ragged]],
):
    # FIXME: do we want to make `remove_bos_eos` configurable as well or
    #        is it always the same post-processing?
    if piece_adapter:
        layers = [
            chain(
                piece_encoder, piece_adapter, with_spans(transformer), remove_bos_eos()
            )
        ]
    else:
        layers = [chain(piece_encoder, with_spans(transformer), remove_bos_eos())]
    refs = {
        "piece_encoder": piece_encoder,
        "transformer": transformer,
    }
    return Model(
        "transformer_model",
        transformer_model_forward,
        init=transformer_model_init,
        layers=layers,
        refs=refs,
    )


def transformer_model_forward(model: Model, docs: List[Doc], is_train: bool):
    Y, backprop_layer = model.layers[0](docs, is_train=is_train)

    def backprop(dY):
        backprop_layer(dY)

        # Return empty list for backprop, since we cannot backprop into piece
        # identifiers.
        return []

    return Y, backprop


def transformer_model_init(model: Model, X: List[Doc] = None, Y=None):
    model.layers[0].initialize(X, Y)
    return model


def _torchscript_encoder(*, model_max_length: int, padding_idx: int):
    return TorchScriptWrapper_v1(
        convert_inputs=partial(
            _convert_inputs,
            max_model_seq_len=model_max_length,
            padding_idx=padding_idx,
        ),
        convert_outputs=_convert_outputs,
    )
