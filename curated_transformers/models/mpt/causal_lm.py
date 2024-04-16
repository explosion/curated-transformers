from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Type, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from ...layers.attention import AttentionMask
from ...layers.cache import KeyValueCache
from ...quantization import Quantizable
from ..hf_hub import FromHF
from ..hf_hub.conversion import state_dict_from_hf, state_dict_to_hf
from ..output import CausalLMOutputWithCache
from ..transformer import TransformerCausalLM
from ._hf import CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS, _config_from_hf, _config_to_hf
from .config import MPTConfig
from .decoder import MPTDecoder

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="MPTCausalLM")


class MPTCausalLM(TransformerCausalLM[MPTConfig], FromHF[MPTConfig], Quantizable):
    """
    `MosaicML MPT`_ causal language model.

    .. _MosaicML MPT: https://www.mosaicml.com/blog/mpt-7b
    """

    def __init__(
        self, config: MPTConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct an MPT causal LM.

        :param config:
            Causal LM configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The causal LM.
        """
        super().__init__(config)

        self.decoder = MPTDecoder(config, device=device)

        # Once we have proper support for tied weights, we will do something like:
        #
        # self.output_embeddings = Linear(
        #    in_features=config.layer.feedforward.hidden_width,
        #    out_features=config.embedding.n_pieces,
        #    bias=False,
        #    device=device,
        # )
        # self.output_embeddings.weights = self.decoder.embeddings.piece_embeddings.weights
        #
        # For now we'll work around this by using the piece embeddings directly.

    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        cache: Optional[List[KeyValueCache]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> CausalLMOutputWithCache[KeyValueCache]:
        # TODO: remove this forward method once we support weight tying.

        decoder_output = self.decoder(
            piece_ids,
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
        )

        assert isinstance(self.decoder.embeddings.piece_embeddings, Embedding)
        output_embeddings = self.decoder.embeddings.piece_embeddings.weight

        logits = F.linear(decoder_output.last_hidden_layer_state, output_embeddings)
        return CausalLMOutputWithCache(
            all_outputs=decoder_output.all_outputs,
            cache=decoder_output.cache,
            logits=logits,
        )

    @classmethod
    def is_supported(cls: Type[Self], config: Dict[str, Any]) -> bool:
        return config.get("model_type") == "mpt"

    @classmethod
    def state_dict_from_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_from_hf(params, CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def state_dict_to_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_to_hf(params, CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def config_from_hf(cls, hf_config: Mapping[str, Any]) -> MPTConfig:
        return _config_from_hf(hf_config)

    @classmethod
    def config_to_hf(cls, curated_config: MPTConfig) -> Mapping[str, Any]:
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

    @classmethod
    def modules_to_not_quantize(cls) -> Set[str]:
        # Ignore the output embedding matrix.
        return {"output_embeddings"}
