from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, Identity, Linear, Module

from .attention import AttentionMask, KeyValueCache, SelfAttention
from .feedforward import PointwiseFeedForward


@dataclass
class EmbeddingLayerNorms:
    """
    Layer normalizations used in a transformer embedding layer.

    By default, all the normalizations are disabled by setting the layer
    normalization to the Torch ``Identity`` module. Therefore, only
    normalizations that are needed have to be set.

    :param embed_output_layer_norm:
        Normalization of the embeddings.
    :param proj_output_layer_norm:
        Normalization of the output of the projection layer.
    """

    embed_output_layer_norm: Module = Identity()
    proj_output_layer_norm: Module = Identity()


@dataclass
class EmbeddingDropouts:
    """
    Dropouts used in a transformer embedding layer.

    By default, all the dropouts are disabled by setting the dropout
    to the Torch ``Identity`` module. Therefore, only dropouts that are
    needed have to be set.

    :param embed_output_dropout:
        Dropout of the embeddings.
    :param proj_output_dropout:
        Dropout of the output of the projection layer.
    """

    embed_output_dropout: Module = Identity()
    proj_output_dropout: Module = Identity()


class TransformerEmbeddings(Module):
    """
    Transformer embeddings layer.

    This is a generic transformer embedding layer. The layer always has piece
    embeddings and can optionally have position embeddings, type embeddings,
    and a projection of embeddings to the model's hidden size.
    """

    def __init__(
        self,
        *,
        dropouts: EmbeddingDropouts,
        embedding_width: int,
        hidden_width: int,
        layer_norms: EmbeddingLayerNorms,
        n_pieces: int,
        n_positions: Optional[int],
        n_types: Optional[int],
        device: Optional[torch.device] = None,
    ):
        """
        Construct an embeddings layer.

        :param dropouts:
            Dropouts to use in the embeddings layer.
        :param embedding_width:
            Width of the embeddings.
        :param hidden_width:
            Hidden width of the transformer. If this width differs from
            ``embedding_width``, a projection layer is added to ensure
            that the output of the embeddings layer has the same width as the
            transformer.
        :param layer_norms:
            Layer norms to use in the embeddings layer.
        :param n_pieces:
            Number of piece embeddings.
        :param n_positions:
            Number of position embeddings. Position embeddings are disabled
            by using ``None``. Position embeddings can be used to inform the
            model of input order.
        :param n_types:
            Number of type embeddings. Type embeddings are disabled by using
            ``None``. Type embeddings can be used to inform the model of the
            spans of different sequences in the input.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()

        self.piece_embeddings = Embedding(
            num_embeddings=n_pieces,
            embedding_dim=embedding_width,
            device=device,
        )

        if n_types is None:
            self.type_embeddings = None
        else:
            self.type_embeddings = Embedding(
                num_embeddings=n_types,
                embedding_dim=embedding_width,
                device=device,
            )

        if n_positions is None:
            self.position_embeddings = None
        else:
            self.position_embeddings = Embedding(
                num_embeddings=n_positions,
                embedding_dim=embedding_width,
                device=device,
            )

        if embedding_width != hidden_width:
            self.projection: Linear = Linear(
                embedding_width,
                hidden_width,
                device=device,
            )
        else:
            self.projection = None  # type: ignore

        self.embed_output_layer_norm = layer_norms.embed_output_layer_norm
        self.proj_output_layer_norm = layer_norms.proj_output_layer_norm

        self.embed_output_dropout = dropouts.embed_output_dropout
        self.proj_output_dropout = dropouts.proj_output_dropout

    def _get_positions(self, x: Tensor) -> Tensor:
        return torch.arange(x.shape[1], device=x.device).expand(1, -1)

    def _get_type_ids(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def forward(
        self,
        piece_ids: Tensor,
        *,
        positions: Optional[Tensor] = None,
        type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply the embedding layer to the piece identifiers.

        :param piece_ids:
            Piece identifiers to embed.

            *Shape:* ``(batch_size, seq_len)``
        :param positions:
            Positional with which to fetch the positional embeddings for the
            sequences.

            *Shape:* ``(batch_size, seq_len)``
        :param type_ids:
            Type identifiers to indicate the spans of different
            sequences in the input. Useful when performing tasks like
            sequence classification and question answering.

            *Shape:* ``(batch_size, seq_len)``
        """

        embeddings = self.piece_embeddings(piece_ids)

        if self.type_embeddings is not None:
            if type_ids is None:
                type_ids = self._get_type_ids(piece_ids)
            embeddings += self.type_embeddings(type_ids)

        if self.position_embeddings is not None:
            if positions is None:
                positions = self._get_positions(piece_ids)
            position_embeddings = self.position_embeddings(positions)
            embeddings += position_embeddings

        embeddings = self.embed_output_layer_norm(embeddings)
        embeddings = self.embed_output_dropout(embeddings)

        if self.projection is not None:
            embeddings = self.projection(embeddings)
            embeddings = self.proj_output_layer_norm(embeddings)
            return self.proj_output_dropout(embeddings)
        else:
            return embeddings


@dataclass
class TransformerLayerNorms:
    """
    Layer normalizations used in a transformer layer.

    By default, all the normalizations are disabled by setting the layer
    normalization to the Torch ``Identity`` module. Therefore, only
    normalizations that are needed have to be set.

    :param attn_input_layer_norm:
        Normalization of the input to the attention layer.
    :param attn_residual_layer_norm:
        Normalization of the output of the attention layer after the
        residual connection.
    :param ffn_input_layer_norm:
        Normalization of the input to the feed-forward layer.
    :param ffn_residual_layer_norm:
        Normalization of the output of the feed-forward layer after the
        residual connection.
    """

    attn_input_layer_norm: Module = Identity()
    attn_residual_layer_norm: Module = Identity()
    ffn_input_layer_norm: Module = Identity()
    ffn_residual_layer_norm: Module = Identity()


@dataclass
class TransformerDropouts:
    """
    Dropouts used in a transformer layer.

    By default, all the dropouts are disabled by setting the dropout
    to the Torch ``Identity`` module. Therefore, only dropouts that are
    needed have to be set.

    :param attn_output_dropout:
        Dropout of the output of the attention layer.
    :param ffn_output_dropout:
        Dropout of the output of the attention layer.
    :param parallel_attn_dropout:
        Dropout after summing the attention and feed-forward layers. Only
        used when parallel attention is enabled.
    """

    attn_output_dropout: Module = Identity()
    ffn_output_dropout: Module = Identity()
    parallel_attn_dropout: Module = Identity()

    @classmethod
    def layer_output_dropouts(cls, p: float) -> "TransformerDropouts":
        """
        Utility method to construct attention and feed-forward layer
        dropouts.

        :param p:
            Dropout probability.
        :returns:
            Dropouts of attention and feed-forward layers set to ``p``.
        """
        return TransformerDropouts(
            attn_output_dropout=Dropout(p), ffn_output_dropout=Dropout(p)
        )

    @classmethod
    def parallel_attention_dropout(cls, p: float) -> "TransformerDropouts":
        """
        Utility method to construct parallel attention dropout.

        :param p:
            Dropout probability.
        :returns:
            Dropouts of parallel attention set to ``p``.
        """
        return TransformerDropouts(parallel_attn_dropout=Dropout(p))


class _TransformerLayer(Module):
    """
    Transformer decoder layer (`Vaswani et al., 2017`_) base class.

    This is a generic transformer layer. :py:class:`DecoderLayer` and
    :py:class:`EncoderLayer` provide specialized encoder/decoder layers.

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    # Don't expose this module, we cannot provide the full forward method
    # without violating the Liskov substitution principle. Ideally, we'd
    # use this module using composition inside the encoder/decoder layers,
    # but we can't without adding a prefix.

    def __init__(
        self,
        *,
        attention_layer: SelfAttention,
        dropouts: TransformerDropouts,
        feed_forward_layer: PointwiseFeedForward,
        layer_norms: TransformerLayerNorms,
        use_parallel_attention: bool,
    ):
        """
        Construct a transformer layer.

        :param attention_layer:
            The attention layer to use in the transformer layer.
        :param dropouts:
            Dropouts to use in the transformer layer.
        :param feed_forward_layer:
            The pointwise feed-forward layer to use in the transformer layer.
        :param layer_norms:
            Layer norms to use in the layer.
        :param use_parallel_attention:
             Use parallel attention.
        """
        super().__init__()

        self.use_parallel_attention = use_parallel_attention

        self.mha = attention_layer

        self.ffn = feed_forward_layer

        self.attn_output_dropout = dropouts.attn_output_dropout
        self.ffn_output_dropout = dropouts.ffn_output_dropout
        self.parallel_attn_dropout = dropouts.parallel_attn_dropout

        self.attn_input_layer_norm = layer_norms.attn_input_layer_norm
        self.attn_residual_layer_norm = layer_norms.attn_residual_layer_norm
        self.ffn_input_layer_norm = layer_norms.ffn_input_layer_norm
        self.ffn_residual_layer_norm = layer_norms.ffn_residual_layer_norm

    def _forward(
        self,
        input: Tensor,
        attention_mask: AttentionMask,
        *,
        cache: Optional[KeyValueCache] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
        use_causal_mask: bool,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply the transformer layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param use_causal_mask:
            Mask out succeeding sequence elements when ``True``.
        :param cache:
            Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up rotary embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Layer output and the key/value cache.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        residual = input

        attn_out, cache = self.mha(
            self.attn_input_layer_norm(input),
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
            use_causal_mask=use_causal_mask,
        )
        attn_out = self.attn_output_dropout(attn_out)

        if self.use_parallel_attention:
            ffn_in = input
        else:
            residual = self.attn_residual_layer_norm(input + attn_out)
            ffn_in = residual

        ffn_out = self.ffn(self.ffn_input_layer_norm(ffn_in))
        ffn_out = self.ffn_output_dropout(ffn_out)

        if self.use_parallel_attention:
            output = self.parallel_attn_dropout(attn_out + ffn_out)
        else:
            output = ffn_out

        return self.ffn_residual_layer_norm(residual + output), cache


class DecoderLayer(_TransformerLayer):
    """
    Transformer decoder layer (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    def forward(
        self,
        input: Tensor,
        attention_mask: AttentionMask,
        *,
        cache: Optional[KeyValueCache] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply the decoder layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param cache:
            Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up rotary embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Layer output and the key/value cache.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return super()._forward(
            input,
            attention_mask=attention_mask,
            cache=cache,
            use_causal_mask=True,
            positions=positions,
            store_cache=store_cache,
        )


class EncoderLayer(_TransformerLayer):
    """
    Transformer encoder layer (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    def forward(
        self,
        input: Tensor,
        attention_mask: AttentionMask,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply the encoder layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :returns:
            Layer output and the key/value cache.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return super()._forward(
            input, attention_mask=attention_mask, use_causal_mask=False
        )
