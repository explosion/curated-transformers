import torch
from typing import Optional
from ..bert.config import BertEmbeddingConfig
from ..embeddings import SinusoidalPositionalEmbedding

from torch import Tensor
from torch.nn import Module, Embedding, Dropout, LayerNorm


class SinusoidalEmbeddings(Module):
    def __init__(self, embedding_config: BertEmbeddingConfig):
        super().__init__()

        self.word_embeddings = Embedding(
            num_embeddings=embedding_config.vocab_size,
            embedding_dim=embedding_config.embedding_width,
        )
        self.layer_norm = LayerNorm(
            embedding_config.embedding_width, eps=embedding_config.layer_norm_eps
        )
        self.dropout = Dropout(p=embedding_config.dropout_prob)

        self.sinusoidal = SinusoidalPositionalEmbedding(
            dim=embedding_config.embedding_width, max_len=10000
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Shapes:
            input_ids, token_type_ids, position_ids - (batch, seq_len)
        """
        embeddings = self.word_embeddings(input_ids.long())
        _, seq_len, dim = embeddings.shape
        position_embeddings = self.sinusoidal(embeddings)
        with torch.no_grad():
            embeddings += position_embeddings

        embeddings = self.layer_norm(embeddings)

        return self.dropout(embeddings)
