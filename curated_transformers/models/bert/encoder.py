from typing import Any, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from ..attention import AttentionMask
from ..hf_hub import FromPretrainedHFModel
from ..output import ModelOutput
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import BertConfig
from .embeddings import BertEmbeddings
from .layer import BertEncoderLayer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="BertEncoder")


class BertEncoder(Module, FromPretrainedHFModel):
    def __init__(self, config: BertConfig, *, device: Optional[torch.device] = None):
        super().__init__()

        self.embeddings = BertEmbeddings(config.embedding, config.layer, device=device)
        self.padding_id = config.padding_id
        self.max_seq_len = config.model_max_length
        self.layers = torch.nn.ModuleList(
            [
                BertEncoderLayer(config.layer, config.attention, device=device)
                for _ in range(config.layer.num_hidden_layers)
            ]
        )

    def _create_attention_mask(self, x: Tensor) -> AttentionMask:
        return AttentionMask(bool_mask=x.ne(self.padding_id))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        """
        Shapes:
            input_ids, attention_mask, token_type_ids - (batch, seq_len)
        """
        if attention_mask is None:
            attention_mask = self._create_attention_mask(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layer_outputs = []
        for layer in self.layers:
            layer_output = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)

        return ModelOutput(
            embedding_output=embeddings, layer_hidden_states=layer_outputs
        )

    @classmethod
    def convert_hf_state_dict(cls, params: Mapping[str, Parameter]):
        return convert_hf_state_dict(params)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config, device=device)
