from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

from .config import BertConfig
from .embeddings import BertEmbeddings
from .layer import BertEncoderLayer
from ..attention import AttentionMask
from ..output import PyTorchTransformerOutput


class BertEncoder(Module):
    def __init__(
        self,
        config: BertConfig,
    ):
        super().__init__()

        self.embeddings = BertEmbeddings(config.embedding, config.layer)
        self.padding_idx = config.padding_idx
        self.max_seq_len = config.model_max_length
        self.layers = torch.nn.ModuleList(
            [
                BertEncoderLayer(config.layer, config.attention)
                for _ in range(config.layer.num_hidden_layers)
            ]
        )

    def _create_attention_mask(self, x: Tensor) -> AttentionMask:
        return AttentionMask(bool_mask=x.ne(self.padding_idx))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> PyTorchTransformerOutput:
        """
        Shapes:
            input_ids, attention_mask, token_type_ids - (batch, seq_len)
        """
        if attention_mask is None:
            attention_mask = self._create_attention_mask(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layer_outputs = []
        for layer in self.layers:
            layer_output = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)

        return PyTorchTransformerOutput(
            embedding_output=embeddings, layer_hidden_states=layer_outputs
        )
