from typing import Any, List, Mapping, Optional, Tuple, Type, TypeVar
from torch import Tensor
from torch.nn import Embedding, LayerNorm, ModuleList, Parameter


from ..hf_hub import FromPretrainedHFModel
from ..module import DecoderModule
from ..attention import AttentionMask, KeyValueCache
from ..output import ModelOutput, ModelOutputWithCache
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import GPTNeoXConfig
from .layer import GPTNeoXDecoderLayer


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="GPTNeoXDecoder")


class GPTNeoXDecoder(DecoderModule, FromPretrainedHFModel):
    """GPT-NeoX (Black et al, 2022) decoder."""

    def __init__(self, config: GPTNeoXConfig) -> None:
        """
        :param config: Model configuration.
        """
        super().__init__()

        self.embeddings = Embedding(
            config.embedding.vocab_size, config.embedding.embedding_width
        )

        self.layers = ModuleList(
            [
                GPTNeoXDecoderLayer(config.layer, config.attention)
                for _ in range(config.layer.num_hidden_layers)
            ]
        )

        self.output_layer_norm = LayerNorm(
            config.layer.hidden_width, config.layer.layer_norm_eps
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        cache: Optional[List[KeyValueCache]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> ModelOutputWithCache[KeyValueCache]:
        """
        Apply the GPT-NeoX decoder to the given piece identifiers.

        :param input_ids: Piece identifiers to apply the decoder to.
        :param attn_mask: Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            in attention.
        :param cache: Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions: Input positions. Positions are needed to
            look up rotary embeddings. Normally, these positions are calculated
            automatically. But if the positions deviate for some reason, they
            can be provided through this argument.
        :param store_cache: Whether to cache the key/value representations for
            future reuse.
        :returns: Decoder representations of the given pieces.

        Shapes:
            input_ids, attention_mask, token_type_ids - (batch, seq_len)
        """
        embeddings = self.embeddings(input_ids)
        layer_output = embeddings

        layer_outputs = []
        new_cache = []
        layer_cache = None
        for layer in self.layers:
            if cache is not None:
                layer_cache = cache[0]
                cache = cache[1:]
            layer_output, new_layer_cache = layer(
                layer_output,
                attention_mask,
                cache=layer_cache,
                store_cache=store_cache,
                positions=positions,
            )
            layer_outputs.append(layer_output)
            if store_cache:
                new_cache.append(new_layer_cache)

        layer_outputs[-1] = self.output_layer_norm(layer_outputs[-1])

        return ModelOutputWithCache(
            embedding_output=embeddings,
            layer_hidden_states=layer_outputs,
            cache=new_cache if store_cache else None,
        )

    @classmethod
    def convert_hf_state_dict(cls, params: Mapping[str, Parameter]):
        return convert_hf_state_dict(cls, params)

    @classmethod
    def from_hf_config(cls: Type[Self], *, hf_config: Any) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config)
