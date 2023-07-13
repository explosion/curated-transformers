from typing import Any, List, Mapping, Optional, Set, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Linear

from ...layers.attention import AttentionMask
from ...layers.cache import KeyValueCache
from ...quantization.quantizable import Quantizable
from ..hf_hub import FromHFHub
from ..module import CausalLMModule
from ..output import CausalLMOutputWithCache
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import FalconConfig
from .decoder import FalconDecoder

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconCausalLM")


class FalconCausalLM(CausalLMModule[KeyValueCache], FromHFHub, Quantizable):
    """
    `Falcon`_ causal language model.

    .. _Falcon: https://arxiv.org/abs/2306.01116
    """

    def __init__(
        self, config: FalconConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a Falcon causal LM.

        :param config:
            Causal LM configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The causal LM.
        """
        super().__init__()

        self.decoder = FalconDecoder(config, device=device)
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
        logits = self.output_embeddings(decoder_output.last_hidden_layer_state)

        return CausalLMOutputWithCache(
            cache=decoder_output.cache,
            embedding_output=decoder_output.embedding_layer,
            layer_hidden_states=decoder_output.all_hidden_layer_states,
            logits=logits,
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

    @classmethod
    def modules_to_not_quantize(cls) -> Set[str]:
        # Ignore the output embedding matrix.
        return {"output_embeddings"}
