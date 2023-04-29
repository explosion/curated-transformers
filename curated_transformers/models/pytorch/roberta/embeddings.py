from typing import Optional
from torch.nn import Module
from torch import Tensor

from ..bert import BertEmbeddings, BertEmbeddingConfig, BertLayerConfig


class RobertaEmbeddings(Module):
    def __init__(
        self,
        embedding_config: BertEmbeddingConfig,
        layer_config: BertLayerConfig,
        *,
        padding_idx: int
    ) -> None:
        super().__init__()

        self.inner = BertEmbeddings(embedding_config, layer_config)
        self.padding_idx = padding_idx

    def _get_position_ids(self, x: Tensor) -> Tensor:
        # We need to generate the position IDs from the
        # input tensor to pass to the embedding layer and
        # handle padding, c.f https://github.com/huggingface/transformers/blob/330247ede2d8265aae9ab0b7a0d1a811c344960d/src/transformers/models/roberta/modeling_roberta.py#L1566

        mask = x.ne(self.padding_idx).int()
        return (mask.cumsum(dim=1) * mask) + self.padding_idx

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
        if position_ids is None:
            position_ids = self._get_position_ids(input_ids)

        return self.inner(input_ids, token_type_ids, position_ids)
