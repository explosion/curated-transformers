from typing import List, Tuple, Callable, cast
from thinc.layers.pytorchwrapper import PyTorchWrapper_v2
from thinc.model import Model
from thinc.types import ArgsKwargs, Ragged
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.util import torch2xp, xp2torch
import torch
from torch import Tensor
from torch.nn import Module
from spacy.util import SimpleFrozenDict

from ..util import all_equal
from .types import ScalarWeightInT, ScalarWeightOutT, ScalarWeightModelT


# From syntaxdot:
# https://github.com/tensordot/syntaxdot/blob/22bd3d43ed2d7fcbef8a6217b01684194fae713f/syntaxdot-transformers/src/scalar_weighting.rs#L62
class ScalarWeight(Module):
    def __init__(self, *, num_layers: int, dropout_prob: float = 0.1):
        super().__init__()

        self.layer_weights = torch.nn.parameter.Parameter(torch.zeros(num_layers))
        self.scale = torch.nn.parameter.Parameter(torch.tensor((1.0,)))
        self.dropout_prob = dropout_prob

    def forward(
        self,
        layer_outputs: Tensor,
    ) -> Tensor:
        """
        Shapes:
            layer_outputs - (seq_len, num_layers, layer_size)

        Returns a weighted tensor of the input with shape (seq_len, layer_size).
        """
        if layer_outputs.shape[1] != self.layer_weights.shape[0]:
            raise ValueError(
                f"Expected {self.layer_weights.shape[0]} layers, got {layer_outputs.shape[1]} instead"
            )

        if self.training:
            dropout_mask = torch.full_like(
                self.layer_weights, 1.0 - self.dropout_prob
            ).bernoulli()
            softmask_mask = (1.0 - dropout_mask) * -10000.0
            layer_weights = self.layer_weights + softmask_mask
        else:
            layer_weights = self.layer_weights

        # Convert the layer weights into a probability distribution and
        # expand dimensions to get shape [1, n_layers, 1].
        layer_weights = layer_weights.softmax(dim=-1).unsqueeze(0).unsqueeze(-1)
        weighted_layers = layer_outputs * layer_weights

        return weighted_layers.sum(dim=-2, keepdim=False) * self.scale


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
