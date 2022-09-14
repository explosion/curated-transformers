from typing import Optional
import torch
from torch import Tensor
from torch.nn import Module

from .config import BertConfig


class BertEmbeddings(Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        self.word_embeddings = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.token_type_embeddings = torch.nn.Embedding(
            num_embeddings=config.type_vocab_size, embedding_dim=config.hidden_size
        )
        self.position_embeddings = torch.nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
        )

        self.layer_norm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)

    def _get_position_ids(self, x: Tensor) -> Tensor:
        return torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape)

    def _get_token_type_ids(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if token_type_ids is None:
            token_type_ids = self._get_token_type_ids(input_ids)
        if position_ids is None:
            position_ids = self._get_position_ids(input_ids)

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embedding_sum = input_embeddings + position_embeddings + token_type_embeddings
        normalized = self.layer_norm(embedding_sum)
        return self.dropout(normalized)
