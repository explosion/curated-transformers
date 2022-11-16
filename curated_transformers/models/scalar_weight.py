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
    layer_hidden_sizes = [x.data.shape[1] for x in X]
    if not all_equal(layer_hidden_sizes):
        raise ValueError(
            f"Not all hidden sizes are equal in input passed to scalar weight"
        )

    Xops = ops.alloc3f(X[0].data.shape[0], len(X), X[0].data.shape[1])
    for i, layer in enumerate(X):
        Xops[:, i, :] = layer.dataXd  # type: ignore
    Xt = xp2torch(Xops, requires_grad=True)

    def convert_from_torch_backward(d_inputs: ArgsKwargs):
        # (seq, num_layers, hidden)
        dt_inputs: Tensor = cast(Tensor, d_inputs.args[0])

        dX = []
        for i in range(dt_inputs.shape[1]):
            dX_layer = dt_inputs[:, i, :]
            dX.append(Ragged(data=torch2xp(dX_layer, ops=ops), lengths=X[0].lengths))
        return dX

    output = ArgsKwargs(args=(Xt,), kwargs={})
    return output, convert_from_torch_backward


def _convert_outputs(
    model: Model, inputs_outputs: Tuple[ScalarWeightInT, Tensor], is_train: bool
) -> Tuple[ScalarWeightOutT, Callable[[Ragged], ArgsKwargs]]:
    X, Yt = inputs_outputs
    Y = Ragged(torch2xp(Yt, ops=model.ops), lengths=X[0].lengths)

    def convert_for_torch_backward(dY: Ragged) -> ArgsKwargs:
        dYt = xp2torch(dY.dataXd)
        return ArgsKwargs(
            args=(Yt,),
            kwargs={"grad_tensors": dYt},
        )

    return Y, convert_for_torch_backward
