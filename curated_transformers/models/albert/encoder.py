from typing import Any, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor

from ...layers.attention import AttentionMask
from ..bert.embeddings import BERTEmbeddings
from ..hf_hub import FromHFHub
from ..module import EncoderModule
from ..output import ModelOutput
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import ALBERTConfig
from .layer_group import ALBERTLayerGroup

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="ALBERTEncoder")


class ALBERTEncoder(EncoderModule, FromHFHub):
    """
    ALBERT (`Lan et al., 2022`_) encoder.

    .. _Lan et al., 2022: https://arxiv.org/abs/1909.11942
    """

    def __init__(self, config: ALBERTConfig, *, device: Optional[torch.device] = None):
        """
        Construct an ALBERT encoder.

        :param config:
            Encoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The encoder.
        """
        super().__init__()

        self.padding_id = config.padding_id
        self.max_seq_len = config.model_max_length
        self.num_hidden_layers = config.layer.num_hidden_layers
        num_hidden_groups = config.layer.num_hidden_groups

        if self.num_hidden_layers % num_hidden_groups != 0:
            raise ValueError(
                f"The number of hidden layers ({self.num_hidden_layers}) in the "
                "ALBERT encoder must be divisable by number of hidden groups "
                f"({num_hidden_groups})"
            )

        self.embeddings = BERTEmbeddings(config.embedding, config.layer, device=device)

        # Parameters are shared by groups of layers.
        self.groups = torch.nn.ModuleList(
            [
                ALBERTLayerGroup(config.layer, config.attention, device=device)
                for _ in range(num_hidden_groups)
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
        if attention_mask is None:
            attention_mask = self._create_attention_mask(input_ids)

        embeddings = self.embeddings(input_ids, token_type_ids, None)
        layer_output = embeddings

        layers_per_group = self.num_hidden_layers // len(self.groups)

        layer_outputs = []
        for group in self.groups:
            for _ in range(layers_per_group):
                layer_output = group(layer_output, attention_mask=attention_mask)
                layer_outputs.append(layer_output)

        return ModelOutput(
            embedding_output=embeddings, layer_hidden_states=layer_outputs
        )

    @classmethod
    def convert_hf_state_dict(cls, params: Mapping[str, Tensor]):
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
