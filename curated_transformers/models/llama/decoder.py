from functools import partial
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, ModuleList

from ...layers.attention import (
    AttentionHeads,
    QkvMode,
    QkvSplitGroupedByKVHeads,
    ScaledDotProductAttention,
    SelfAttention,
)
from ...layers.embeddings import QueryKeyRotaryEmbeddings
from ...layers.feedforward import PointwiseFeedForward
from ...layers.normalization import RMSNorm
from ...layers.transformer import (
    DecoderLayer,
    EmbeddingDropouts,
    EmbeddingLayerNorms,
    TransformerDropouts,
    TransformerEmbeddings,
    TransformerLayerNorms,
)
from ..hf_hub import FromHF
from ..hf_hub.conversion import state_dict_from_hf, state_dict_to_hf
from ..transformer import TransformerDecoder
from ._hf import DECODER_HF_PARAM_KEY_TRANSFORMS, _config_from_hf, _config_to_hf
from .config import LlamaConfig

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="LlamaDecoder")


class LlamaDecoder(TransformerDecoder[LlamaConfig], FromHF[LlamaConfig]):
    """
    Llama (`Touvron et al., 2023 [a]`_, `Touvron et al., 2023 [b]`_) decoder.

    .. _Touvron et al., 2023 [a]: https://arxiv.org/abs/2302.13971
    .. _Touvron et al., 2023 [b]: https://arxiv.org/abs/2307.09288
    """

    def __init__(
        self, config: LlamaConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a Llama decoder.

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
        n_query_heads = config.layer.attention.n_query_heads
        attention_heads = AttentionHeads.key_value_broadcast(
            n_query_heads=n_query_heads,
            n_key_value_heads=config.layer.attention.n_key_value_heads,
            qkv_split=QkvSplitGroupedByKVHeads(),
        )
        layer_norm = partial(
            RMSNorm,
            hidden_width,
            eps=config.layer.layer_norm_eps,
            device=device,
        )
        if config.layer.attention.rotary_embeddings is None:
            raise ValueError(
                "Llama attention config does not contain rotary embedding parameters"
            )
        self.layers = ModuleList(
            [
                DecoderLayer(
                    attention_layer=SelfAttention(
                        attention_heads=attention_heads,
                        attention_scorer=ScaledDotProductAttention(
                            dropout_prob=config.layer.attention.dropout_prob,
                            linear_biases=None,
                        ),
                        hidden_width=hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=QueryKeyRotaryEmbeddings(
                            fraction=config.layer.attention.rotary_embeddings.rotary_fraction,
                            base=config.layer.attention.rotary_embeddings.rotary_base,
                            head_width=hidden_width // n_query_heads,
                        ),
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

        self.output_layer_norm = RMSNorm(
            hidden_width, eps=config.layer.layer_norm_eps, device=device
        )

    @classmethod
    def is_supported(cls: Type[Self], config: Dict[str, Any]) -> bool:
        return config.get("model_type") == "llama"

    @classmethod
    def state_dict_from_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_from_hf(params, DECODER_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def state_dict_to_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_to_hf(params, DECODER_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def config_from_hf(cls, hf_config: Mapping[str, Any]) -> LlamaConfig:
        return _config_from_hf(hf_config)

    @classmethod
    def config_to_hf(cls, curated_config: LlamaConfig) -> Mapping[str, Any]:
        return _config_to_hf(cls, curated_config)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = cls.config_from_hf(hf_config)
        return cls(config, device=device)
