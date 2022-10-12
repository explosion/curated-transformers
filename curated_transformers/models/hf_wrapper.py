from typing import List
from functools import partial

from spacy.tokens import Doc
from thinc.api import Model, get_current_ops
from thinc.api import xp2torch, PyTorchWrapper_v2
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import ArgsKwargs, Floats2d, Ints1d
from thinc.util import torch2xp

from .._compat import transformers, has_hf_transformers
from ..util import registry
from .hf_util import convert_hf_pretrained_model_parameters, SupportedHfTransformersT
from .output import TransformerModelOutput


def build_hf_transformer_encoder_v1(
    encoder: SupportedHfTransformersT,
    *,
    mixed_precision: bool = False,
    grad_scaler_config: dict = {},
) -> Model[List[Ints1d], List[Floats2d]]:
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


def build_hf_encoder_loader_v1(
    *,
    name: str,
    revision: str = "main",
):
    def load(model: Model, X: List[Doc] = None, Y=None):
        if not has_hf_transformers:
            raise ValueError("requires 🤗 transformers")

        global transformers
        from transformers import AutoModel

        encoder = model.shims[0]._model

        hf_model = AutoModel.from_pretrained(name, revision=revision)
        params = convert_hf_pretrained_model_parameters(hf_model)
        encoder.load_state_dict(params)

        return model

    return load
