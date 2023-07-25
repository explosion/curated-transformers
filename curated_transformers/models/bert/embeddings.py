from typing import Optional

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module

from ..config import TransformerEmbeddingLayerConfig, TransformerLayerConfig


class BERTEmbeddings(Module):
    """
    BERT (`Devlin et al., 2018`_) embedding layer.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    def __init__(
        self,
        embedding_config: TransformerEmbeddingLayerConfig,
        layer_config: TransformerLayerConfig,
        *,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        if embedding_config.type_vocab_size is None:
            raise ValueError(
                "BERT embedding config does not contain type vocabulary size"
            )
        elif embedding_config.max_position_embeddings is None:
            raise ValueError(
                "BERT embedding config does not contain max position embeddings length"
            )

        self.word_embeddings = Embedding(
            num_embeddings=embedding_config.vocab_size,
            embedding_dim=embedding_config.embedding_width,
            device=device,
        )
        self.token_type_embeddings: Embedding = Embedding(
            num_embeddings=embedding_config.type_vocab_size,
            embedding_dim=embedding_config.embedding_width,
            device=device,
        )
        self.position_embeddings: Embedding = Embedding(
            num_embeddings=embedding_config.max_position_embeddings,
            embedding_dim=embedding_config.embedding_width,
            device=device,
        )

        if embedding_config.embedding_width != layer_config.feedforward.hidden_width:
            self.projection: Linear = Linear(
                embedding_config.embedding_width,
                layer_config.feedforward.hidden_width,
                device=device,
            )
        else:
            self.projection = None  # type: ignore

        self.layer_norm: LayerNorm = LayerNorm(
            embedding_config.embedding_width,
            eps=embedding_config.layer_norm_eps,
            device=device,
        )
        self.dropout: Dropout = Dropout(p=embedding_config.dropout_prob)

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
        """
        Apply the BERT embedding layer to the piece identifiers.

        :param input_ids:
            Piece identifiers to embed.

            *Shape:* ``(batch_size, seq_len)``
        :param token_type_ids:
            Token type identifiers to indicate the spans of different
            sequences in the input. Useful when performing tasks like
            sequence classification and question answering.

            *Shape:* ``(batch_size, seq_len)``
        :param position_ids:
            Positional identifiers with which to fetch the positional
            embeddings for the sequences.

            *Shape:* ``(batch_size, seq_len)``
        """
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
