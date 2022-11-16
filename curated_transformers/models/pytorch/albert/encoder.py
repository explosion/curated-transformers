from typing import Optional
import torch
from torch.nn import Linear, Module
from torch import Tensor

from ..attention import AttentionMask
from ..bert.embeddings import BertEmbeddings
from ...output import PyTorchTransformerOutput
from .config import AlbertConfig
from .layer_group import AlbertLayerGroup


class AlbertEncoder(Module):
    def __init__(
        self,
        config: AlbertConfig,
    ):
        super().__init__()

        self.padding_idx = config.padding_idx
        self.max_seq_len = config.model_max_length
        self.num_hidden_layers = config.layer.num_hidden_layers
        num_hidden_groups = config.layer.num_hidden_groups

        if self.num_hidden_layers % num_hidden_groups != 0:
            raise ValueError(
                f"Number of hidden layers ({self.num_hidden_layers}) must be divisable by number of hidden groups ({num_hidden_groups})"
            )

        self.embeddings = BertEmbeddings(config.embedding, config.layer)

        # Parameters are shared by groups of layers.
        self.groups = torch.nn.ModuleList(
            [
                AlbertLayerGroup(config.layer, config.attention)
                for _ in range(num_hidden_groups)
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
            input_ids, token_type_ids - (batch, seq_len)
        """
        if attention_mask is None:
            attention_mask = self._create_attention_mask(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layers_per_group = self.num_hidden_layers // len(self.groups)

        layer_outputs = []
        for i in range(self.num_hidden_layers):
            layer_output = self.groups[i // layers_per_group](
                layer_output, attn_mask=attention_mask
            )
            layer_outputs.append(layer_output)

        return PyTorchTransformerOutput(
            embedding_output=embeddings, layer_hidden_states=layer_outputs
        )
