from functools import partial
from typing import Any, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, LayerNorm, ModuleList

from ...layers.attention import (
    AttentionHeads,
    AttentionLinearBiases,
    QkvMode,
    SelfAttention,
)
from ...layers.embeddings import QueryKeyRotaryEmbeddings
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
from .config import FalconConfig
from .layer import OldFalconDecoderLayer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconDecoder")


class FalconDecoder(TransformerDecoder[FalconConfig], FromHFHub):
    """
    Falcon (`Penedo et al., 2019`_) decoder.

    .. _Penedo et al., 2019: https://arxiv.org/abs/2306.01116
    """

    def __init__(
        self, config: FalconConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a Falcon decoder.

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

        if config.new_decoder_architecture:
            decoder_layer = partial(
                self._create_new_decoder_architecture_layer, config, device
            )
        else:
            decoder_layer = partial(
                self._create_old_decoder_architecture_layer, config, device
            )

        self.layers = ModuleList(
            [decoder_layer() for _ in range(config.layer.n_hidden_layers)]
        )

        self.output_layer_norm = LayerNorm(
            config.layer.feedforward.hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )

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

    def _create_old_decoder_architecture_layer(
        self, config: FalconConfig, device: Optional[torch.device]
    ):
        return OldFalconDecoderLayer(config.layer, device=device)

    def _create_new_decoder_architecture_layer(
        self, config: FalconConfig, device: Optional[torch.device]
    ):
        if config.layer.attention.rotary_embeddings is None:
            raise ValueError(
                "Falcon attention config does not contain rotary embedding parameters"
            )

        hidden_width = config.layer.feedforward.hidden_width
        layer_norm = partial(
            LayerNorm,
            hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )
        n_attention_heads = config.layer.attention.n_query_heads
        attention_biases = (
            AttentionLinearBiases(
                n_attention_heads=config.layer.attention.n_query_heads,
                is_causal=True,
                is_inverted=True,
            )
            if config.layer.attention.use_alibi
            else None
        )
        # Rotary embeddings are disabled when using ALiBi.
        rotary_embeds = (
            QueryKeyRotaryEmbeddings(
                fraction=config.layer.attention.rotary_embeddings.rotary_fraction,
                base=config.layer.attention.rotary_embeddings.rotary_base,
                head_width=hidden_width // n_attention_heads,
            )
            if not config.layer.attention.use_alibi
            else None
        )
        return DecoderLayer(
            attention_layer=SelfAttention(
                attention_biases=attention_biases,
                attention_heads=AttentionHeads.key_value_broadcast(
                    n_query_heads=n_attention_heads,
                    n_key_value_heads=config.layer.attention.n_key_value_heads,
                ),
                dropout_prob=config.layer.attention.dropout_prob,
                hidden_width=hidden_width,
                qkv_mode=QkvMode.MERGED_SPLIT_AFTER,
                rotary_embeds=rotary_embeds,
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
            dropouts=TransformerDropouts.parallel_attention_dropout(
                config.layer.dropout_prob
            ),
            layer_norms=TransformerLayerNorms(
                attn_input_layer_norm=layer_norm(),
                ffn_input_layer_norm=layer_norm(),
            ),
            # The new decoder uses parallel attention unconditionally.
            use_parallel_attention=True,
        )
