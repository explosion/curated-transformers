from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .activations import GeluFast, GeluNew


class PointwiseFeedForward(Module):
    """Point-wise feed-forward layer (Vaswani et al., 2017).

    This layer is applied pointwise, meaning that the same
    transformation is applied to each sequence element. This
    transformation is: g(xW_1 + b_1)W_2 + b_2, where W_1 and b_1 transform
    the input to an intermediate width, g is a non-linear activation
    function and W_2 and b_2 transform the output of the activation back to
    the input width.

    hidden_act (str): the activation function to apply, one of: "relu",
        "gelu" or "gelu_new" (default: "gelu").
    hidden_width (int): the input and output width of the layer.
        (default: 768)
    intermediate_width (int): the width of the projection to which the
        non-linearity is applied. (default: 3072)
    """

    def __init__(
        self,
        *,
        hidden_act: str = "gelu",
        hidden_width: int = 768,
        intermediate_width: int = 3072,
        use_bias: bool,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.intermediate = Linear(
            hidden_width, intermediate_width, bias=use_bias, device=device
        )
        self.output = Linear(
            intermediate_width, hidden_width, bias=use_bias, device=device
        )
        if hidden_act == "relu":
            self.activation = torch.nn.ReLU()  # type: ignore
        elif hidden_act == "gelu":
            self.activation = torch.nn.GELU()  # type: ignore
        elif hidden_act == "gelu_new":
            # Ideally, we would use torch.nn.GELU(approximate="tanh"). However,
            # the differences between that and the manual Torch implementation
            # are large enough to fail tests comparing output to HF
            # transformers.
            self.activation = GeluNew()  # type: ignore
        elif hidden_act == "gelu_fast":
            self.activation = GeluFast()  # type: ignore
        else:
            supported_activations = ("relu", "gelu", "gelu_new", "gelu_fast")
            raise ValueError(
                f"Invalid activation function `{hidden_act}` for point-wise feed-forward "
                f"network. Supported functions: {supported_activations}"
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the point-wise feedforward layer to the input.

        Shapes:
            x - (batch, seq_len, width)
            output - (batch, seq_len, width)
        """
        out = self.intermediate(x)
        out = self.activation(out)
        out = self.output(out)
        return out
