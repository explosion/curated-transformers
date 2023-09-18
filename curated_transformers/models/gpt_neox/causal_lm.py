from typing import Any, Mapping, Optional, Set, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Linear

from ...quantization import Quantizable
from ..hf_hub import FromHFHub
from ..transformer import TransformerCausalLM
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import GPTNeoXConfig
from .decoder import GPTNeoXDecoder

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="GPTNeoXCausalLM")


class GPTNeoXCausalLM(TransformerCausalLM[GPTNeoXConfig], FromHFHub, Quantizable):
    """
    GPT-NeoX (`Black et al., 2022`_) causal language model.

    .. _Black et al., 2022: https://arxiv.org/abs/2204.06745
    """

    def __init__(
        self, config: GPTNeoXConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a GPT-NeoX causal LM.

        :param config:
            Causal LM configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The causal LM.
        """
        super().__init__(config)

        self.decoder = GPTNeoXDecoder(config, device=device)
        self.output_embeddings = Linear(
            in_features=config.layer.feedforward.hidden_width,
            out_features=config.embedding.n_pieces,
            bias=False,
            device=device,
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
