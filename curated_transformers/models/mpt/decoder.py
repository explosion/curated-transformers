from typing import Any, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, ModuleList

from ...layers.attention import (
    AttentionHeads,
    AttentionLinearBiases,
    QkvMode,
    QkvSplitGroupedByHead,
    SelfAttention,
)
from ...layers.feedforward import PointwiseFeedForward
from ...layers.transformer import (
    DecoderLayer,
    EmbeddingDropouts,
    EmbeddingLayerNorms,
    TransformerDropouts,
    TransformerEmbeddings,
    TransformerLayerNorms,
)
from ..hf_hub import FromHFHub
from ..transformer import TransformerDecoder
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import MPTConfig

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="MPTDecoder")


class MPTDecoder(TransformerDecoder[MPTConfig], FromHFHub):
    """
    `MosaicML MPT`_ decoder.

    .. _MosaicML MPT: https://www.mosaicml.com/blog/mpt-7b
    """

    def __init__(
        self, config: MPTConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct an MPT decoder.

        :param config:
            Decoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The decoder.
        """
        super().__init__(config)

        self.embeddings = TransformerEmbeddings(
            dropouts=EmbeddingDropouts(
                embed_output_dropout=Dropout(config.embedding.dropout_prob)
            ),
            embedding_width=config.embedding.embedding_width,
            hidden_width=config.layer.feedforward.hidden_width,
            layer_norms=EmbeddingLayerNorms(),
            n_pieces=config.embedding.n_pieces,
            n_positions=None,
            n_types=None,
            device=device,
        )

        hidden_width = config.layer.feedforward.hidden_width
        n_attention_heads = config.layer.attention.n_query_heads

        def layer_norm():
            layer_norm = LayerNorm(
                hidden_width, config.layer.layer_norm_eps, device=device
            )
            if not config.layer.feedforward.use_bias:
                layer_norm.bias = None
            return layer_norm

        attention_biases = AttentionLinearBiases(
            n_attention_heads=config.layer.attention.n_query_heads,
            is_causal=True,
            is_inverted=False,
        )

        self.layers = ModuleList(
            [
                DecoderLayer(
                    attention_layer=SelfAttention(
                        attention_biases=attention_biases,
                        attention_heads=AttentionHeads.uniform(
                            n_attention_heads, QkvSplitGroupedByHead()
                        ),
                        dropout_prob=config.layer.attention.dropout_prob,
                        hidden_width=hidden_width,
                        qkv_mode=QkvMode.MERGED_SPLIT_AFTER,
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
                        attn_input_layer_norm=layer_norm(),
                        ffn_input_layer_norm=layer_norm(),
                    ),
                    use_parallel_attention=config.layer.attention.use_parallel_attention,
                )
                for _ in range(config.layer.n_hidden_layers)
            ]
        )

        self.output_layer_norm = layer_norm()

    @classmethod
    def convert_hf_state_dict(cls, params: Mapping[str, Tensor]):
        return convert_hf_state_dict(cls, params)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config, device=device)
