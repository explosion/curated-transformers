from typing import List, Tuple, Callable
import torch
from torch import Tensor
from torch.nn import Module


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
