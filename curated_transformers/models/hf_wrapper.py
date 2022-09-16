import re
import torch

from typing import Tuple, List, Dict
from functools import partial

from thinc.api import Model, torch2xp, get_current_ops
from thinc.api import xp2torch, PyTorchWrapper_v2
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import ArgsKwargs, Floats2d, Floats3d, Ints1d

from .roberta import RobertaEncoder, RobertaConfig
from .._compat import transformers, has_hf_transformers
from .hf_util import convert_hf_pretrained_model_parameters


def build_hf_transformer_encoder_v1(
    hf_model_name: str,
    hf_model_revision: str = "main",
    mixed_precision: bool = False,
    grad_scaler_config: dict = {},
) -> Model[List[Ints1d], List[Floats2d]]:
    if not has_hf_transformers:
        raise ValueError("requires 🤗 transformers")

    encoder = encoder_from_pretrained_hf_model(hf_model_name, hf_model_revision)

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
    return model


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
            f"span window size of '{max_seq_len}' exceeds maximum allowed sequence length of '{max_model_seq_len}'"
        )

    # Transform the list of strided spans to a padded array.
    Xt = ops.xp.full((len(X), max_seq_len), padding_idx)
    for i in range(len(X)):
        span = X[i]
        span_len = span.shape[0]
        Xt[i, :span_len] = span
    Xt = xp2torch(Xt)

    def convert_from_torch_backward(d_inputs):
        # FIXME: d_inputs is an ellipsis here - bug?
        # No gradients for the inputs.
        return [ops.alloc1f(x.shape[0]) for x in X]

    output = ArgsKwargs(args=(Xt,), kwargs={})
    return output, convert_from_torch_backward


def _convert_outputs(model, inputs_outputs, is_train):
    X, model_outputs = inputs_outputs

    # Strip the outputs for the padding timesteps
    # and return the outputs from the last Transformer layer.
    last_layer_output = model_outputs.last_hidden_state

    X_lens = [x.shape[0] for x in X]
    Yt = [last_layer_output[i, :len, :] for i, len in enumerate(X_lens)]
    Y = [torch2xp(yt) for yt in Yt]

    def convert_for_torch_backward(dY: List[Floats2d]):
        dYt = [xp2torch(y) for y in dY]
        return ArgsKwargs(
            args=(Yt,),
            kwargs={"grad_tensors": dYt},
        )

    return Y, convert_for_torch_backward


def encoder_from_pretrained_hf_model(
    model_name: str, model_revision: str = "main"
) -> RobertaEncoder:
    global transformers
    from transformers import AutoModel, AutoTokenizer

    hf_model = AutoModel.from_pretrained(model_name, revision=model_revision)
    hf_config = hf_model.config
    model_tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)

    config = RobertaConfig(
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        attention_probs_dropout_prob=hf_config.attention_probs_dropout_prob,
        hidden_dropout_prob=hf_config.hidden_dropout_prob,
        hidden_act=hf_config.hidden_act,
        vocab_size=hf_config.vocab_size,
        type_vocab_size=hf_config.type_vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        model_max_length=model_tokenizer.model_max_length,
        layer_norm_eps=hf_config.layer_norm_eps,
        padding_idx=hf_config.pad_token_id,
    )
    encoder = RobertaEncoder(config)

    params = convert_hf_pretrained_model_parameters(hf_model)
    encoder.load_state_dict(params)
    return encoder
