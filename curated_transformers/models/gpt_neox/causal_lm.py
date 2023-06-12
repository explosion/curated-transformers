from typing import Any, List, Mapping, Optional, Type, TypeVar
import torch
from torch import Tensor
from torch.nn import Linear

from ..attention import AttentionMask, KeyValueCache
from ._hf import convert_hf_config, convert_hf_state_dict
from ..hf_hub import FromPretrainedHFModel
from ..module import CausalLMModule
from ..output import CausalLMOutputWithCache
from .config import GPTNeoXConfig
from .decoder import GPTNeoXDecoder
from ..util.serde import DeserializationParamBucket


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="GPTNeoXCausalLM")


class GPTNeoXCausalLM(CausalLMModule[KeyValueCache], FromPretrainedHFModel):
    """GPT-NeoX (Black et al, 2022) causal language model."""

    def __init__(
        self, config: GPTNeoXConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        :param config: Model configuration.
        :param device: Device on which the module is to be initialized.
        """
        super().__init__()

        self.decoder = GPTNeoXDecoder(config, device=device)
        self.output_embeddings = Linear(
            in_features=config.layer.hidden_width,
            out_features=config.embedding.vocab_size,
            bias=False,
            device=device,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        cache: Optional[List[KeyValueCache]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> CausalLMOutputWithCache[KeyValueCache]:
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

    def deserialization_param_buckets(self) -> List[DeserializationParamBucket]:
        return []

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
