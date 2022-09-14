from typing import List
from dataclasses import dataclass
from torch import Tensor


@dataclass
class TransformerEncoderOutput:
    layer_outputs: List[Tensor]  # [batch, seq, model_hidden]
    embedding_sum: Tensor  # [batch, seq, emb_dim]

    @property
    def last_hidden_output(self) -> Tensor:
        return self.layer_outputs[len(self.layer_outputs) - 1]
