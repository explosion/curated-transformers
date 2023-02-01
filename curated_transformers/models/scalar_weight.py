from typing import List, Tuple, Callable, cast
from thinc.layers.pytorchwrapper import PyTorchWrapper_v2
from thinc.model import Model
from thinc.types import ArgsKwargs, Ragged
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.util import torch2xp, xp2torch
from torch import Tensor
from spacy.util import SimpleFrozenDict

from ..util import all_equal
from .types import ScalarWeightInT, ScalarWeightOutT, ScalarWeightModelT
from .pytorch.scalar_weight import ScalarWeight


def build_scalar_weight_v1(
    *,
    num_layers: int,
    dropout_prob: float = 0.1,
    mixed_precision: bool = False,
    grad_scaler_config: dict = SimpleFrozenDict(),
) -> ScalarWeightModelT:
    """Construct a model that accepts a list of transformer layer
    outputs and returns a weighted representation of the same.

    num_layers (int):
        Number of transformer hidden layers.
    dropout_prob (float):
        Dropout probability.
    mixed_precision (bool):
        Use mixed-precision training.
    grad_scaler_config (dict):
        Configuration passed to the PyTorch gradient scaler.
    """
    if isinstance(grad_scaler_config, SimpleFrozenDict):
        # Create a new, mutable dict instance.
        grad_scaler_config = {}

    if "enabled" not in grad_scaler_config:
        grad_scaler_config["enabled"] = mixed_precision

    # Increment number of layers by one to include the embedding layer.
    scalar_weighting_layer = ScalarWeight(
        num_layers=num_layers + 1, dropout_prob=dropout_prob
    )

    model = PyTorchWrapper_v2(
        scalar_weighting_layer,
        convert_inputs=_convert_inputs,
        convert_outputs=_convert_outputs,
        mixed_precision=mixed_precision,
        grad_scaler=PyTorchGradScaler(**grad_scaler_config),
    )

    return model


def _convert_inputs(
    model: Model, X: ScalarWeightInT, is_train: bool = False
) -> Tuple[ArgsKwargs, Callable[[ArgsKwargs], List[Ragged]]]:
    ops = model.ops

    batch_size = len(X)
    seq_lens = [x[0].data.shape[0] for x in X]
    max_seq_len = max(seq_lens)
    num_layers = [len(x) for x in X]
    if not all_equal(num_layers):
        raise ValueError(f"Not all inputs have the same number of layers")
    layer_widths = [layer.data.shape[1] for x in X for layer in x]
    if not all_equal(layer_widths):
        raise ValueError(
            f"Not all hidden widths are equal in input passed to scalar weight"
        )

    # [batch_size, max_seq_len, num_layers, layer_width]
    Xops = ops.alloc4f(batch_size, max_seq_len, num_layers[0], layer_widths[0])
    for doc_idx, layers in enumerate(X):
        seq_len = seq_lens[doc_idx]
        for layer_idx, data in enumerate(layers):
            Xops[doc_idx, :seq_len, layer_idx, :] = data.dataXd  # type: ignore
    Xt = xp2torch(Xops, requires_grad=True)

    def convert_from_torch_backward(d_inputs: ArgsKwargs):
        # [batch, seq, num_layers, hidden]
        dt_inputs: Tensor = cast(Tensor, d_inputs.args[0])

        dX = []
        for doc_idx in range(dt_inputs.shape[0]):
            seq_len = seq_lens[doc_idx]
            lengths = X[doc_idx][0].lengths
            dX_layers = []
            for layer_idx in range(dt_inputs.shape[2]):
                dX_layer = dt_inputs[doc_idx, :seq_len, layer_idx, :]
                dX_layers.append(
                    Ragged(data=torch2xp(dX_layer, ops=ops), lengths=lengths)
                )
            dX.append(dX_layers)
        return dX

    output = ArgsKwargs(args=(Xt,), kwargs={})
    return output, convert_from_torch_backward


def _convert_outputs(
    model: Model, inputs_outputs: Tuple[ScalarWeightInT, Tensor], is_train: bool
) -> Tuple[ScalarWeightOutT, Callable[[List[Ragged]], ArgsKwargs]]:
    ops = model.ops
    X, Yt = inputs_outputs

    Y = []
    for doc_idx in range(len(X)):
        seq_len = X[doc_idx][0].data.shape[0]
        lengths = X[doc_idx][0].lengths
        # [batch, seq, hidden]
        Y_layer = Yt[doc_idx, :seq_len, :]  # type: ignore
        Y.append(Ragged(torch2xp(Y_layer, ops=model.ops), lengths=lengths))

    def convert_for_torch_backward(dY: List[Ragged]) -> ArgsKwargs:
        max_seq_len = max(y.data.shape[0] for y in dY)
        width = dY[0].data.shape[1]

        dYt_ops = ops.alloc3f(len(dY), max_seq_len, width)
        for doc_idx in range(len(dY)):
            seq_len = dY[doc_idx].data.shape[0]
            dYt_ops[doc_idx, :seq_len, :] = dY[doc_idx].dataXd  # type: ignore

        dYt = xp2torch(dYt_ops)
        return ArgsKwargs(
            args=(Yt,),
            kwargs={"grad_tensors": dYt},
        )

    return Y, convert_for_torch_backward
