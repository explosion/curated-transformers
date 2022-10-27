from typing import List
from pathlib import Path
from functools import partial
from spacy.tokens import Span, Doc
from spacy.util import SimpleFrozenDict
from thinc.api import (
    Model,
    PyTorchWrapper_v2,
    get_current_ops,
    xp2torch,
    torch2xp,
    get_torch_default_device,
)
from thinc.layers import chain
from thinc.model import Model
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import ArgsKwargs, Floats2d, Ints1d, Ragged
import torch


from .hf_util import SupportedEncoders, convert_pretrained_model_for_encoder
from .albert import AlbertConfig, AlbertEncoder
from .bert import BertConfig, BertEncoder
from .output import TransformerModelOutput
from .remove_eos_bos import remove_bos_eos
from .roberta import RobertaConfig, RobertaEncoder
from .torchscript_wrapper import TorchScriptWrapper_v1
from .with_non_ws_tokens import with_non_ws_tokens
from ..tokenization.bbpe_encoder import build_byte_bpe_encoder
from ..tokenization.sentencepiece_adapters import build_xlmr_adapter
from ..tokenization.sentencepiece_encoder import build_sentencepiece_encoder
from ..tokenization.wordpiece_encoder import build_wordpiece_encoder


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
    torchscript: bool = False,
    mixed_precision: bool = False,
    grad_scaler_config: dict = SimpleFrozenDict(),
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
        transformer = _pytorch_encoder(
            encoder,
            mixed_precision=mixed_precision,
            grad_scaler_config=grad_scaler_config,
        )

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
    torchscript: bool = False,
    mixed_precision: bool = False,
    grad_scaler_config: dict = SimpleFrozenDict(),
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
        transformer = _pytorch_encoder(
            encoder,
            mixed_precision=mixed_precision,
            grad_scaler_config=grad_scaler_config,
        )

    piece_encoder = build_wordpiece_encoder()

    return build_transformer_model_v1(
        with_spans=with_spans,
        piece_encoder=piece_encoder,
        transformer=transformer,
    )


def build_roberta_transformer_model_v1(
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
    torchscript: bool = False,
    mixed_precision: bool = False,
    grad_scaler_config: dict = SimpleFrozenDict(),
):
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
        encoder = RobertaEncoder(config)
        transformer = _pytorch_encoder(
            encoder,
            mixed_precision=mixed_precision,
            grad_scaler_config=grad_scaler_config,
        )

    piece_encoder = build_byte_bpe_encoder()

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
    torchscript: bool = False,
    mixed_precision: bool = False,
    grad_scaler_config: dict = SimpleFrozenDict(),
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
        encoder = RobertaEncoder(config)
        transformer = _pytorch_encoder(
            encoder,
            mixed_precision=mixed_precision,
            grad_scaler_config=grad_scaler_config,
        )

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
            with_non_ws_tokens(chain(
                piece_encoder, piece_adapter, with_spans(transformer), remove_bos_eos()
            ))
        ]
    else:
        layers = [with_non_ws_tokens(chain(piece_encoder, with_spans(transformer), remove_bos_eos()))]
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


def _pytorch_encoder(
    encoder: SupportedEncoders,
    *,
    mixed_precision: bool = False,
    grad_scaler_config: dict = SimpleFrozenDict(),
) -> Model[List[Ints1d], List[Floats2d]]:
    if isinstance(grad_scaler_config, SimpleFrozenDict):
        # Crate a new, mutable dict instance.
        grad_scaler_config = {}

    if "enabled" not in grad_scaler_config:
        grad_scaler_config["enabled"] = mixed_precision

    model = PyTorchWrapper_v2(
        encoder,
        convert_inputs=partial(
            _convert_inputs,
            max_model_seq_len=encoder.max_seq_len,
            padding_idx=encoder.padding_idx,
        ),
        convert_outputs=_convert_outputs,
        mixed_precision=mixed_precision,
        grad_scaler=PyTorchGradScaler(**grad_scaler_config),
    )

    # Actual value initialized by the parent Pipe instance.
    model.attrs["_all_layer_outputs"] = True

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


def _convert_inputs(
    model: Model,
    X: List[Ints1d],
    is_train: bool = False,
    max_model_seq_len=512,
    padding_idx: int = 1,
):
    ops = get_current_ops()

    max_seq_len = max(x.size for x in X)
    if max_seq_len > max_model_seq_len:
        raise ValueError(
            f"Span window size of '{max_seq_len}' exceeds maximum allowed sequence length of '{max_model_seq_len}'"
        )

    # Transform the list of strided spans to a padded array.
    Xt = ops.xp.full((len(X), max_seq_len), padding_idx)
    for i in range(len(X)):
        span = X[i]
        span_len = span.shape[0]
        Xt[i, :span_len] = span
    Xt = xp2torch(Xt)

    def convert_from_torch_backward(d_inputs):
        # No gradients for the inputs.
        return [ops.alloc1f(x.shape[0]) for x in X]

    output = ArgsKwargs(args=(Xt,), kwargs={})
    return output, convert_from_torch_backward


def _convert_outputs(model, inputs_outputs, is_train):
    model_inputs, model_outputs = inputs_outputs
    ops = get_current_ops()
    all_layer_outputs: bool = model.attrs["_all_layer_outputs"]

    # Strip the outputs for the padding timesteps
    # while preserving all the outputs (and their order).
    input_lens = [x.shape[0] for x in model_inputs]
    if all_layer_outputs:
        Yt = [
            [output[i, :len, :] for output in model_outputs.all_outputs]
            for i, len in enumerate(input_lens)
        ]
    else:
        Yt = [
            [model_outputs.all_outputs[-1][i, :len, :]]
            for i, len in enumerate(input_lens)
        ]

    Y = [[torch2xp(layer, ops=ops) for layer in output] for output in Yt]
    output = TransformerModelOutput(outputs=Y)
    output.last_layer_only = not all_layer_outputs

    def convert_for_torch_backward(dY: List[List[Floats2d]]):
        Yt_flat = [y for inner in Yt for y in inner]
        dYt_flat = [xp2torch(y) for inner in dY for y in inner]
        assert len(Yt_flat) == len(dYt_flat)

        return ArgsKwargs(
            args=(Yt_flat,),
            kwargs={"grad_tensors": dYt_flat},
        )

    return output, convert_for_torch_backward


def build_pytorch_checkpoint_loader_v1(*, path: Path):
    def load(model: Model, X: List[Doc] = None, Y=None):
        encoder = model.shims[0]._model
        device = get_torch_default_device()
        params = torch.load(path, map_location=device)
        params = convert_pretrained_model_for_encoder(encoder, params)
        encoder.load_state_dict(params)
        encoder.to(device)

    return load
