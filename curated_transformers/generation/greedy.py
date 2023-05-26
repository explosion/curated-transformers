from typing import Generic, Iterator, List, Optional, Tuple
import torch
from torch import Tensor

from ..models.attention import AttentionMask, CacheT
from ..models.module import CausalLMModule
from .state import GenerationState


class GreedyGenerator(Generic[CacheT]):
    model: CausalLMModule[CacheT]

    def __init__(self, model: CausalLMModule[CacheT]) -> None:
        self.model = model

    def __call__(
        self, *, attention_mask: Tensor, ids: Tensor, eos_id: int
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        return self.decode(attention_mask=attention_mask, ids=ids, eos_id=eos_id)

    def decode(
        self, *, attention_mask: Tensor, ids: Tensor, eos_id: int
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        self.model.eval()

        cache: Optional[List[CacheT]] = None
        state = GenerationState(attention_mask=attention_mask, cache=cache)

        while True:
            with torch.no_grad():
                output = self.model(
                    ids,
                    attention_mask=AttentionMask(state.attention_mask),
                    cache=state.cache,
                    store_cache=True,
                    positions=state.positions,
                )
            best = output.logits.argmax(-1)
            ids = best[:, -1:]

            completed = ids.view(-1) == eos_id
            ids = ids[completed.logical_not()]
            seq_ids = state.seq_ids
            state.step(cache=output.cache, completed=completed)

            if state.is_finished:
                return

            yield seq_ids, ids
