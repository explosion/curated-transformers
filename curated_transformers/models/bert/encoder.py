from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

from .config import BertConfig
from .embeddings import BertEmbeddings
from .layer import BertEncoderLayer
from ..output import TransformerEncoderOutput


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
                BertEncoderLayer(config.layer, config.attention)
                for _ in range(config.layer.num_hidden_layers)
            ]
        )

    def _create_attention_mask(self, x: Tensor) -> Tensor:
        return x.ne(self.padding_idx).int()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> TransformerEncoderOutput:
        """
        Shapes:
            input_ids, token_type_ids - (batch, seq_len)

        `attn_mask` indicates elements to attend to with `1` (and `0` otherwise)

        Returns a tuple of consisting of a list of tensors from each Transformer
        layer and the sum of the input and positional embeddings.
        """
        if attention_mask is None:
            attention_mask = self._create_attention_mask(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layer_outputs = []
        for layer in self.layers:
            layer_output = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)

        return TransformerEncoderOutput(
            embedding_output=embeddings, layer_hidden_states=layer_outputs
        )
