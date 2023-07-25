from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from ..bert import BERTEmbeddings
from ..config import TransformerEmbeddingLayerConfig, TransformerLayerConfig


class RoBERTaEmbeddings(Module):
    """
    RoBERTa (`Liu et al., 2019`_) embedding layer.

    .. _Liu et al., 2019: https://arxiv.org/abs/1907.11692
    """

    def __init__(
        self,
        embedding_config: TransformerEmbeddingLayerConfig,
        layer_config: TransformerLayerConfig,
        *,
        padding_id: int,
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()

        self.inner = BERTEmbeddings(embedding_config, layer_config, device=device)
        self.padding_id = padding_id

    def _get_position_ids(self, x: Tensor) -> Tensor:
        # We need to generate the position IDs from the
        # input tensor to pass to the embedding layer and
        # handle padding, c.f https://github.com/huggingface/transformers/blob/330247ede2d8265aae9ab0b7a0d1a811c344960d/src/transformers/models/roberta/modeling_roberta.py#L1566

        mask = x.ne(self.padding_id).int()
        return (mask.cumsum(dim=1) * mask) + self.padding_id

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply the RoBERTa embedding layer to the piece identifiers.

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
        if position_ids is None:
            position_ids = self._get_position_ids(input_ids)

        return self.inner(input_ids, token_type_ids, position_ids)
