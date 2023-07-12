from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from .attention import (
    AttentionMask,
    QkvHeadSharing,
    QkvMode,
    RotaryEmbeddingConfig,
    SelfAttention,
)
from .feedforward import PointwiseFeedForward


class EncoderLayer(Module):
    """
    Transformer encoder layer (Vaswani et al., 2017).
    """

    def __init__(
        self,
        *,
        attention_dropout: float,
        hidden_act: str,
        hidden_dropout: float,
        hidden_width: int,
        intermediate_width: int,
        layer_norm_eps: float,
        num_attention_heads: int,
        qkv_head_sharing: QkvHeadSharing,
        qkv_mode: QkvMode,
        rotary_embeds: Optional[RotaryEmbeddingConfig],
        use_bias: bool,
        device: Optional[torch.device] = None
    ):
        """
        Construct an encoder layer.

        :param attention_dropout:
            Dropout probabilty for self-attention.
        :param hidden_act:
            Activation used by the feed-forward layers.
            Applied on the intermediate representation.
            See :class:`curated_transformers.layers.feedforward.PointwiseFeedForward`
            for supported activations.
        :param hidden_dropout:
            Dropout probabilty to apply after hidden layers.
        :param hidden_width:
            Hidden width of the transformer.
        :param intermediate_width:
            Intermediate width in the feed-forward layer.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param num_attention_heads:
            Number of self-attention heads.
        :param qkv_head_sharing:
            Head sharing in query, key and value.
        :param qkv_mode:
            Handling mode for query, key and value.
        :param rotary_embeds:
            Configuration for rotary embeddings. Rotary embeddings will not
            be used when set to ``None``.
        :param use_bias:
            Use biases for linear layers.
        :param device:
            Device on which the module is to be initialized.
        """

        super().__init__()

        self.mha = SelfAttention(
            dropout_prob=attention_dropout,
            hidden_width=hidden_width,
            qkv_head_sharing=qkv_head_sharing,
            num_attention_heads=num_attention_heads,
            qkv_mode=qkv_mode,
            rotary_embeds=rotary_embeds,
            use_bias=use_bias,
            device=device,
        )
        self.attn_output_layernorm = torch.nn.LayerNorm(
            hidden_width, eps=layer_norm_eps, device=device
        )
        self.attn_output_dropout = torch.nn.Dropout(p=hidden_dropout)
        self.ffn = PointwiseFeedForward(
            hidden_act=hidden_act,
            hidden_width=hidden_width,
            intermediate_width=intermediate_width,
            use_bias=True,
            use_gate=False,
            device=device,
        )
        self.ffn_output_layernorm = torch.nn.LayerNorm(
            hidden_width, eps=layer_norm_eps, device=device
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=hidden_dropout)

    def forward(self, input: Tensor, attention_mask: AttentionMask) -> Tensor:
        """
        Apply the encoder layer to the input.

        :param input:
            Embeddings to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        """
        attn_out, _ = self.mha(input, attention_mask)
        attn_out = self.attn_output_dropout(attn_out)
        attn_out = self.attn_output_layernorm(input + attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)
        ffn_out = self.ffn_output_layernorm(attn_out + ffn_out)

        return ffn_out
