from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module


# From syntaxdot:
# https://github.com/tensordot/syntaxdot/blob/22bd3d43ed2d7fcbef8a6217b01684194fae713f/syntaxdot-transformers/src/scalar_weighting.rs#L62
class ScalarWeight(Module):
    """
    Scalar weighting (`Peters et al., 2018`_).

    .. _Peters et al., 2018 : https://aclanthology.org/N18-1202/
    """

    def __init__(
        self,
        *,
        num_layers: int,
        dropout_prob: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Construct a scalar weighting module.

        :param num_layers:
            Number of hidden layers.
        :param dropout_prob:
            Dropout applied to the layer weights.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()

        self.layer_weights = torch.nn.parameter.Parameter(
            torch.zeros(num_layers, device=device)
        )
        self.scale = torch.nn.parameter.Parameter(torch.tensor((1.0,), device=device))
        self.dropout_prob = dropout_prob

    def forward(
        self,
        layer_outputs: Tensor,
    ) -> Tensor:
        """
        Apply scalar weighting to the input.

        :param layer_outputs:
            Outputs of the hidden layers.

            *Shape:* ``(batch_size, seq_len, num_layers, width)``
        :returns:
            Weighted tensor of the layer outputs.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        if layer_outputs.shape[2] != self.layer_weights.shape[0]:
            raise ValueError(
                f"Mismatching number of input layers - got '{layer_outputs.shape[1]}', expected '{self.layer_weights.shape[0]}'"
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
        # expand dimensions to get shape [1, 1, n_layers, 1].
        layer_weights = (
            layer_weights.softmax(dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )
        weighted_layers = layer_outputs * layer_weights

        return weighted_layers.sum(dim=-2, keepdim=False) * self.scale
