from typing import Optional
import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module

from .config import BertEmbeddingConfig, BertLayerConfig


class BertEmbeddings(Module):
    def __init__(
        self, embedding_config: BertEmbeddingConfig, layer_config: BertLayerConfig
    ) -> None:
        super().__init__()

        self.word_embeddings = Embedding(
            num_embeddings=embedding_config.vocab_size,
            embedding_dim=embedding_config.embedding_dim,
        )
        self.token_type_embeddings = Embedding(
            num_embeddings=embedding_config.type_vocab_size,
            embedding_dim=embedding_config.embedding_dim,
        )
        self.position_embeddings = Embedding(
            num_embeddings=embedding_config.max_position_embeddings,
            embedding_dim=embedding_config.embedding_dim,
        )

        if embedding_config.embedding_dim != layer_config.hidden_size:
            self.projection = Linear(
                embedding_config.embedding_dim, layer_config.hidden_size
            )
        else:
            self.projection = None  # type: ignore

        self.layer_norm = LayerNorm(
            embedding_config.embedding_dim, eps=embedding_config.layer_norm_eps
        )
        self.dropout = Dropout(p=embedding_config.dropout_prob)

    def _get_position_ids(self, x: Tensor) -> Tensor:
        return torch.arange(x.shape[1], device=x.device).expand(1, -1)

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

        embedding_sum = input_embeddings
        embedding_sum += token_type_embeddings
        embedding_sum += position_embeddings
        normalized = self.layer_norm(embedding_sum)
        embeddings = self.dropout(normalized)

        if self.projection is not None:
            return self.projection(embeddings)
        else:
            return embeddings
