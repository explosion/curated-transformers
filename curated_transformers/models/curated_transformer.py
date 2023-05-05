from typing import Generic, Optional, TypeVar
from torch import Tensor
from torch.nn import Module


from .albert.encoder import AlbertEncoder
from .bert.encoder import BertEncoder
from .roberta.encoder import RobertaEncoder

from .attention import AttentionMask
from .output import PyTorchTransformerOutput

CuratedEncoderT = TypeVar("CuratedEncoderT", AlbertEncoder, BertEncoder, RobertaEncoder)


class CuratedTransformer(Generic[CuratedEncoderT], Module):
    """Simple wrapper for encoders. Currently only used to add a predictable
    prefix (curated_encoder) to encoders."""

    def __init__(self, encoder: CuratedEncoderT) -> None:
        super().__init__()

        # Type ignore, because TorchScript does not allow Module
        # as a class variable type.
        self.curated_encoder = encoder  # type: ignore[var-annotated]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[AttentionMask] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> PyTorchTransformerOutput:
        """
        Shapes:
            input_ids, attention_mask, token_type_ids - (batch, seq_len)
        """
        return self.curated_encoder.forward(input_ids, attention_mask, token_type_ids)
