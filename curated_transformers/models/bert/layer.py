import math
from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

from ..attention import ScaledDotProductAttention
from .config import BertConfig


# https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention
class BertSelfAttention(Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.model_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"model dimension '{self.model_dim}' not divisible by number of heads '{self.num_heads}'"
            )

        self.dims_per_head = self.model_dim // self.num_heads
        self.attention = ScaledDotProductAttention(
            dropout_prob=config.attention_probs_dropout_prob
        )
        self.query = torch.nn.Linear(self.model_dim, self.model_dim)
        self.key = torch.nn.Linear(self.model_dim, self.model_dim)
        self.value = torch.nn.Linear(self.model_dim, self.model_dim)
        self.output = torch.nn.Linear(self.model_dim, self.model_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, emd_dim)
            output - (batch, head, seq_len, dims_per_head)
        """
        batch_size, seq_len, model_dim = x.size()
        return x.view(
            batch_size, seq_len, self.num_heads, self.dims_per_head
        ).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, head, seq_len, dims_per_head)
            output - (batch, seq_len, emd_dim)
        """
        batch_size, head, seq_len, model_dim = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_len, head * model_dim)
        )

    def forward(
        self, k: Tensor, q: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Shapes:
            k, q, v - (batch, seq_len, model_dim)
            attn_mask - (batch, seq_len)

        `attn_mask` indicates elements to attend to with `1` (and `0` otherwise)
        """

        k = self.key(k)
        q = self.query(q)
        v = self.value(v)

        # (batch, head, seq_len, dims_per_head)
        k = self._split_heads(k)
        q = self._split_heads(q)
        v = self._split_heads(v)

        if attn_mask is not None:
            if attn_mask.dim() != 2:
                raise ValueError(
                    f"attention mask dim mismatch, expected '2' but received {attn_mask.dim()}"
                )
            batch, seq_len = attn_mask.size()
            attn_mask = attn_mask.contiguous().view(batch, 1, 1, seq_len)

        # (batch, seq_len, model_dim)
        attn = self._combine_heads(self.attention(k, q, v, attn_mask))
        out = self.output(attn)

        return out


class BertFeedForward(Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.intermediate = torch.nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.output = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        if config.hidden_act == "relu":
            self.activation = torch.nn.ReLU()  # type: ignore
        elif config.hidden_act == "gelu":
            self.activation = torch.nn.GELU()  # type: ignore
        else:
            raise ValueError(f"unsupported activation function '{config.hidden_act}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
        """
        out = self.intermediate(x)
        out = self.activation(out)
        out = self.output(out)
        return out


class BertEncoderLayer(Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.mha = BertSelfAttention(config)
        self.attn_output_layernorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attn_output_dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)
        self.ffn = BertFeedForward(config)
        self.ffn_output_layernorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.ffn_output_dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
            mask - (batch, seq_len)

        `mask` indicates elements to be masked with values of `1`
        """
        attn_out = self.mha(x, x, x, mask)
        attn_out = self.attn_output_dropout(attn_out)
        attn_out = self.attn_output_layernorm(x + attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_output_dropout(ffn_out)
        ffn_out = self.ffn_output_layernorm(attn_out + ffn_out)

        return ffn_out
