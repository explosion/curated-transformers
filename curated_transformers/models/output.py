from typing import List
from dataclasses import dataclass
from torch import Tensor


@dataclass
class TransformerEncoderOutput:
    # The first element is the output of the embedding layer with shape [batch, seq, emb_dim].
    # The rest of the elements are the hidden states of each encoder layer respectively [batch, seq, model_hidden].
    all_outputs: List[Tensor]

    def __init__(
        self, *, embedding_output: Tensor, layer_hidden_states: List[Tensor]
    ) -> None:
        self.all_outputs = [embedding_output]
        self.all_outputs.extend(layer_hidden_states)

    @property
    def embedding_output(self) -> Tensor:
        return self.all_outputs[0]

    def layer_hidden_state(self, idx: int) -> Tensor:
        """'idx' must be in the range [0, num_hidden_layers)"""
        if 0 <= idx < len(self.all_outputs) - 1:
            return self.all_outputs[idx + 1]
        else:
            raise ValueError(
                f"index must be >= 0 and < {len(self.all_outputs) - 1}, got {idx}"
            )

    @property
    def last_hidden_state(self) -> Tensor:
        return self.all_outputs[len(self.all_outputs) - 1]
