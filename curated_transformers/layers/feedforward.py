from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .activations import Activation


class PointwiseFeedForward(Module):
    """
    Point-wise feed-forward layer (`Vaswani et al., 2017`_).

    This layer is applied pointwise, meaning that the same
    transformation is applied to each sequence element. This
    transformation is:

    .. math::
        g(xW_1 + b_1)W_2 + b_2

    :math:`W_1` and :math:`b_1` transform the input to an
    intermediate width, :math:`g` is a non-linear activation
    function and :math:`W_2` and :math:`b_2` transform the
    output of the activation back to the input width.

    Gated Linear Units (`Dauphin et al., 2016`_; `Shazeer, 2020`_) are also
    supported. Gating applies the following transformation:

    .. math::
        (g(xW_g + b_g) * (xW_1 + b_1))W_2 + b_2

    :math:`W_g` and :math:`b_g` are the affine transformation for the gate.

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    .. _Dauphin et al., 2016: https://arxiv.org/abs/1612.08083
    .. _Shazeer, 2020: https://arxiv.org/abs/2002.05202
    """

    gate: Optional[Linear]

    def __init__(
        self,
        *,
        activation: Module,
        hidden_width: int,
        intermediate_width: int,
        use_bias: bool,
        use_gate: bool,
        device: Optional[torch.device] = None,
    ):
        """
        Construct a pointwise feed-forward layer module.

        :param activation:
            Activation used by the pointwise feed-forward layers. The hidden
            input shape must be the same as the output shape (as is typical
            for elementwise activations).
        :param hidden_width:
            The input and output width of the layer.
        :param intermediate_width:
            The width of the projection to which the non-linearity is applied.
        :param use_bias:
            Use biases for linear layers.
        :param use_gate:
            Use Gated Linear Units.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()

        self.intermediate = Linear(
            hidden_width, intermediate_width, bias=use_bias, device=device
        )
        if use_gate:
            self.gate = Linear(
                hidden_width, intermediate_width, bias=use_bias, device=device
            )
        else:
            self.gate = None
        self.output = Linear(
            intermediate_width, hidden_width, bias=use_bias, device=device
        )
        self.activation = activation

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply the point-wise feed-forward layer to the input.

        :param input:
            Input.

            *Shape:* ``(batch_size, seq_len, width)``
        :returns:
            Layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        if self.gate is None:
            return self.output(self.activation(self.intermediate(input)))
        else:
            return self.output(
                self.activation(self.gate(input)) * self.intermediate(input)
            )
