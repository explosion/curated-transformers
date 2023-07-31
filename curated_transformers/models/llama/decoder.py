from functools import partial
from typing import Any, Mapping, Optional, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, ModuleList

from ...layers.attention import AttentionHeads, QkvMode, SelfAttention
from ...layers.embeddings import QueryKeyRotaryEmbeddings
from ...layers.feedforward import PointwiseFeedForward
from ...layers.normalization import RMSNorm
from ...layers.transformer import (
    DecoderLayer,
    TransformerDropouts,
    TransformerLayerNorms,
)
from ..hf_hub import FromHFHub
from ..transformer import TransformerDecoder
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import LLaMAConfig

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="LLaMADecoder")


class LLaMADecoder(TransformerDecoder, FromHFHub):
    """
    LLaMa (`Touvron et al., 2023 [a]`_, `Touvron et al., 2023 [b]`_) decoder.

    .. _Touvron et al., 2023 [a]: https://arxiv.org/abs/2302.13971
    .. _Touvron et al., 2023 [b]: https://arxiv.org/abs/2307.09288
    """

    def __init__(
        self, config: LLaMAConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a LLaMA decoder.

        :param config:
            Decoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The decoder.
        """
        super().__init__()

        self.embeddings = Embedding(
            config.embedding.vocab_size, config.embedding.embedding_width, device=device
        )
        self.dropout = Dropout(p=config.embedding.dropout_prob)

        hidden_width = config.layer.feedforward.hidden_width
        num_query_heads = config.layer.attention.num_query_heads
        attention_heads = AttentionHeads.key_value_broadcast(
            num_query_heads=num_query_heads,
            num_key_value_heads=config.layer.attention.num_key_value_heads,
        )
        layer_norm = partial(
            RMSNorm,
            hidden_width,
            eps=config.layer.layer_norm_eps,
            device=device,
        )
        if config.layer.attention.rotary_embeddings is None:
            raise ValueError(
                "LLaMA attention config does not contain rotary embedding parameters"
            )
        self.layers = ModuleList(
            [
                DecoderLayer(
                    attention_layer=SelfAttention(
                        attention_heads=attention_heads,
                        dropout_prob=config.layer.attention.dropout_prob,
                        hidden_width=hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=QueryKeyRotaryEmbeddings(
                            fraction=config.layer.attention.rotary_embeddings.rotary_fraction,
                            base=config.layer.attention.rotary_embeddings.rotary_base,
                            dims_per_head=hidden_width // num_query_heads,
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
                    parallel_attention=config.layer.attention.parallel_attention,
                )
                for _ in range(config.layer.num_hidden_layers)
            ]
        )

        self.output_layer_norm = RMSNorm(
            hidden_width, eps=config.layer.layer_norm_eps, device=device
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
