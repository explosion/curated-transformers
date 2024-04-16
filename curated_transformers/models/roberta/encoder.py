from functools import partial
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm

from ...layers.attention import (
    AttentionHeads,
    QkvMode,
    QkvSplitGroupedByKVHeads,
    ScaledDotProductAttention,
    SelfAttention,
)
from ...layers.feedforward import PointwiseFeedForward
from ...layers.transformer import (
    EmbeddingDropouts,
    EmbeddingLayerNorms,
    EncoderLayer,
    TransformerDropouts,
    TransformerLayerNorms,
)
from ..hf_hub import FromHF
from ..hf_hub.conversion import state_dict_from_hf, state_dict_to_hf
from ..transformer import TransformerEncoder
from ._hf import HF_PARAM_KEY_TRANSFORMS, _config_from_hf, _config_to_hf
from .config import RoBERTaConfig
from .embeddings import RoBERTaEmbeddings

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="RoBERTaEncoder")


class RoBERTaEncoder(TransformerEncoder[RoBERTaConfig], FromHF[RoBERTaConfig]):
    """
    RoBERTa (`Liu et al., 2019`_) encoder.

    .. _Liu et al., 2019: https://arxiv.org/abs/1907.11692
    """

    def __init__(self, config: RoBERTaConfig, *, device: Optional[torch.device] = None):
        """
        Construct a RoBERTa encoder.

        :param config:
            Encoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The encoder.
        """
        super().__init__(config)

        self.embeddings = RoBERTaEmbeddings(
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
            padding_id=config.padding_id,
        )
        self.max_seq_len = config.model_max_length

        hidden_width = config.layer.feedforward.hidden_width
        layer_norm = partial(
            LayerNorm,
            hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )
        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    attention_layer=SelfAttention(
                        attention_heads=AttentionHeads.uniform(
                            config.layer.attention.n_query_heads,
                            QkvSplitGroupedByKVHeads(),
                        ),
                        attention_scorer=ScaledDotProductAttention(
                            dropout_prob=config.layer.attention.dropout_prob,
                            linear_biases=None,
                        ),
                        hidden_width=hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=None,
                        use_bias=config.layer.attention.use_bias,
                        device=device,
                    ),
                    feed_forward_layer=PointwiseFeedForward(
                        activation=config.layer.feedforward.activation.module(),
                        hidden_width=hidden_width,
                        intermediate_width=config.layer.feedforward.intermediate_width,
                        use_bias=config.layer.feedforward.use_bias,
                        use_gate=config.layer.feedforward.use_gate,
                        device=device,
                    ),
                    dropouts=TransformerDropouts.layer_output_dropouts(
                        config.layer.dropout_prob
                    ),
                    layer_norms=TransformerLayerNorms(
                        attn_residual_layer_norm=layer_norm(),
                        ffn_residual_layer_norm=layer_norm(),
                    ),
                    use_parallel_attention=config.layer.attention.use_parallel_attention,
                )
                for _ in range(config.layer.n_hidden_layers)
            ]
        )

    @classmethod
    def is_supported(cls: Type[Self], config: Dict[str, Any]) -> bool:
        return config.get("model_type") == "roberta"

    @classmethod
    def state_dict_from_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_from_hf(params, HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def state_dict_to_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_to_hf(params, HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def config_from_hf(cls, hf_config: Mapping[str, Any]) -> RoBERTaConfig:
        return _config_from_hf(hf_config)

    @classmethod
    def config_to_hf(cls, curated_config: RoBERTaConfig) -> Mapping[str, Any]:
        return _config_to_hf(curated_config)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = cls.config_from_hf(hf_config)
        return cls(config, device=device)
