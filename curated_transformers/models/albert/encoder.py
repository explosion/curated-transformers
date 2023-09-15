from typing import Any, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm

from ...layers.attention import AttentionMask
from ...layers.transformer import (
    EmbeddingDropouts,
    EmbeddingLayerNorms,
    TransformerEmbeddings,
)
from ..hf_hub import FromHFHub
from ..module import EncoderModule
from ..output import ModelOutput
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import ALBERTConfig
from .layer_group import ALBERTLayerGroup

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="ALBERTEncoder")


class ALBERTEncoder(EncoderModule[ALBERTConfig], FromHFHub):
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
        super().__init__(config)

        self.max_seq_len = config.model_max_length
        self.n_hidden_layers = config.layer.n_hidden_layers
        n_hidden_groups = config.layer.n_hidden_groups

        if self.n_hidden_layers % n_hidden_groups != 0:
            raise ValueError(
                f"The number of hidden layers ({self.n_hidden_layers}) in the "
                "ALBERT encoder must be divisable by number of hidden groups "
                f"({n_hidden_groups})"
            )

        self.embeddings = TransformerEmbeddings(
            dropouts=EmbeddingDropouts(
                embed_output_dropout=Dropout(config.embedding.dropout_prob)
            ),
            embedding_width=config.embedding.embedding_width,
            hidden_width=config.layer.feedforward.hidden_width,
            layer_norms=EmbeddingLayerNorms(
                embed_output_layer_norm=LayerNorm(
                    config.embedding.embedding_width, config.embedding.layer_norm_eps
                )
            ),
            n_pieces=config.embedding.n_pieces,
            n_positions=config.embedding.n_positions,
            n_types=config.embedding.n_types,
            device=device,
        )

        # Parameters are shared by groups of layers.
        self.groups = torch.nn.ModuleList(
            [
                ALBERTLayerGroup(config.layer, device=device)
                for _ in range(n_hidden_groups)
            ]
        )

    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        type_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> ModelOutput:
        embeddings = self.embeddings(piece_ids, positions=positions, type_ids=type_ids)
        layer_output = embeddings

        layers_per_group = self.n_hidden_layers // len(self.groups)

        layer_outputs = []
        for group in self.groups:
            for _ in range(layers_per_group):
                layer_output, _ = group(layer_output, attention_mask=attention_mask)
                layer_outputs.append(layer_output)

        return ModelOutput(all_outputs=[embeddings, *layer_outputs])

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
