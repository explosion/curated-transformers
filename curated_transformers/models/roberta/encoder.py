from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.nn import Module
from torch import Tensor

from ..bert.layer import BertEncoderLayer
from ..output import TransformerEncoderOutput
from .embeddings import RobertaEmbeddings


class RobertaEncoder(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_heads: int,
        n_layers: int,
        attn_dropout: float,
        hidden_dropout: float,
        hidden_activation: str,
        max_pos_embeddings: int,
        vocab_size: int,
        max_seq_len: int,
        *,
        layer_norm_eps: float = 1e-5,
        padding_idx: int = 0,
        type_vocab_size: int = 0,
    ):
        super().__init__()

        self.embeddings = RobertaEmbeddings(
            embedding_dim=hidden_size,
            word_vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            max_pos_embeddings=max_pos_embeddings,
            padding_idx=padding_idx,
            layer_norm_eps=layer_norm_eps,
            dropout=hidden_dropout,
        )
        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len
        self.layers = torch.nn.ModuleList(
            [
                BertEncoderLayer(
                    hidden_size,
                    intermediate_size,
                    n_heads,
                    activation=hidden_activation,
                    attn_dropout=attn_dropout,
                    hidden_dropout=hidden_dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(n_layers)
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
            layer_outputs=layer_outputs, embedding_sum=embeddings
        )
