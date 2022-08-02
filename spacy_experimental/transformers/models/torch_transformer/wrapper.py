from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

from .components import EncoderLayer, SinusoidalPositionalEmbedding


class TransformerEncoder(Module):
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
        *,
        learnable_pos_embeddings=False,
        layer_norm_eps: float = 1e-5,
        padding_idx: int = 0
    ):
        super().__init__()

        self.input_embeddings = torch.nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx
        )
        self.padding_idx = padding_idx
        if learnable_pos_embeddings:
            self.pos_embeddings = torch.nn.Embedding(num_embeddings=max_pos_embeddings, embedding_dim=hidden_size)  # type: ignore
        else:
            self.pos_embeddings = SinusoidalPositionalEmbedding(hidden_size, max_pos_embeddings)  # type: ignore

        self.emb_dropout = torch.nn.Dropout(p=hidden_dropout)
        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(
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

    def _create_mask(self, x: Tensor) -> Tensor:
        return x.eq(self.padding_idx).int()

    def forward(self, input: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Shapes:
            input - (batch, seq_len)

        `attn_mask` indicates elements to be masked with values of `1`
        """
        if not mask:
            mask = self._create_mask(input)

        emb = self.input_embeddings(input)
        pos = self.pos_embeddings(input)

        x = emb + pos
        out = self.emb_dropout(x)

        for layer in self.layers:
            out = layer(out, mask)

        return out
