from typing import Any, List, Mapping, Optional, Type, TypeVar
from torch import Tensor
from torch.nn import Linear, Parameter

from ..attention import AttentionMask, KeyValueCache
from ._hf import convert_hf_config, convert_hf_state_dict
from ..hf_hub import FromPretrainedHFModel
from ..module import CausalLMModule
from ..output import CausalLMOutputWithCache, ModelOutput
from .config import GPTNeoXConfig
from .decoder import GPTNeoXDecoder


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="GPTNeoXCausalLM")


class GPTNeoXCausalLM(CausalLMModule, FromPretrainedHFModel):
    """GPT-NeoX (Black et al, 2022) causal language model."""

    def __init__(self, config: GPTNeoXConfig) -> None:
        """
        :param config: Model configuration.
        """
        super().__init__()

        self.decoder = GPTNeoXDecoder(config)
        self.output_embeddings = Linear(
            in_features=config.layer.hidden_width,
            out_features=config.embedding.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        cache: Optional[List[KeyValueCache]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> CausalLMOutputWithCache[KeyValueCache]:
        """
        Apply the GPT-NeoX causal language model to the given piece identifiers.

        :param input_ids: Piece identifiers to apply the language model to.
        :param attention_mask: Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param cache: Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions: Input positions. Positions are needed to
            look up rotary embeddings. Normally, these positions are calculated
            automatically. But if the positions deviate for some reason, they
            can be provided through this argument.
        :param store_cache: Whether to cache the key/value representations for
            future reuse.
        :returns: Decoder representations of the given pieces and logits of
            the predicted token distribution.

        Shapes:
            input_ids, attention_mask, positions - (batch, seq_len)
        """
        decoder_output = self.decoder(
            input_ids,
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
        )
        logits = self.output_embeddings(decoder_output.last_hidden_layer_states)

        return CausalLMOutputWithCache(
            cache=decoder_output.cache,
            embedding_output=decoder_output.embedding_layer,
            layer_hidden_states=decoder_output.all_hidden_layer_states,
            logits=logits,
        )

    @classmethod
    def convert_hf_state_dict(cls, params: Mapping[str, Parameter]):
        return convert_hf_state_dict(cls, params)

    @classmethod
    def from_hf_config(cls: Type[Self], *, hf_config: Any) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config)
