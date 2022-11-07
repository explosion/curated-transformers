from typing import Optional

import torch
from torch.nn import Module, TransformerEncoderLayer
from torch import Tensor

from .config import BertConfig
from .embeddings import BertEmbeddings
from ..activations import _get_activation
from ..attention import AttentionMask
from ..output import PyTorchTransformerOutput


class BertEncoder(Module):
    def __init__(
        self,
        config: BertConfig,
    ):
        super().__init__()

        self.embeddings = BertEmbeddings(config.embedding)
        self.padding_idx = config.padding_idx
        self.max_seq_len = config.model_max_length
        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    config.layer.hidden_size,
                    config.attention.num_attention_heads,
                    config.layer.intermediate_size,
                    config.layer.dropout_prob,
                    _get_activation(config.layer.hidden_act),
                    config.layer.layer_norm_eps,
                    batch_first=True,
                )
                for _ in range(config.layer.num_hidden_layers)
            ]
        )

    def _create_attention_mask(self, x: Tensor) -> AttentionMask:
        return AttentionMask(bool_mask=x.ne(self.padding_idx))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> PyTorchTransformerOutput:
        """
        Shapes:
            input_ids, token_type_ids - (batch, seq_len)

        `attention_mask` indicates elements to attend to with `True` (and `False` otherwise)

        Returns a tuple of consisting of a list of tensors from each Transformer
        layer and the sum of the input and positional embeddings.
        """
        if attention_mask is None:
            attention_mask = self._create_attention_mask(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layer_outputs = []
        for layer in self.layers:
            layer_output = layer(
                layer_output,
                src_key_padding_mask=attention_mask.bool_mask.logical_not(),
            )
            layer_outputs.append(layer_output)

        return PyTorchTransformerOutput(
            embedding_output=embeddings, layer_hidden_states=layer_outputs
        )
