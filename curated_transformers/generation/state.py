from typing import Generic, List, Optional
import torch
from torch import Tensor

from ..models.attention import CacheT


class GeneratorState(Generic[CacheT]):
    """Generator state."""

    attention_mask: Tensor
    cache: Optional[List[CacheT]]
    positions: Tensor
    seq_ids: Tensor

    def __init__(
        self,
        *,
        attention_mask: Tensor,
        cache: Optional[List[CacheT]],
    ) -> None:
        self.attention_mask = attention_mask
        self.positions = attention_mask.int().cumsum(-1) - 1
        self.cache = cache
        self.seq_ids = torch.arange(
            0, self.attention_mask.size(0), device=attention_mask.device
        )

    @property
    def is_finished(self):
        return len(self.seq_ids) == 0

    def step(self, *, cache: List[CacheT], completed: Optional[Tensor] = None):
        """
        Step the generation state.

        :param cache:
            Model cache from the last model call.
        :param completed:
            Tensor indicating which sequences are completed. These sequences
            are removed from the generation state.
        """
        self.cache = cache

        if completed is not None:
            self._remove_completed(completed)

        self.positions = self.positions.max(-1, keepdim=True).values + 1
        self.attention_mask = torch.cat(
            [
                self.attention_mask,
                torch.full(
                    (self.attention_mask.size(0), 1),
                    True,
                    device=self.attention_mask.device,
                ),
            ],
            dim=-1,
        )

    def _remove_completed(self, completed: Tensor):
        """Remove completed sequences.

        :param completed:
            Tensor indicating for the active sequences whether they are completed.
        """
        not_completed = completed.logical_not()
        self.attention_mask = self.attention_mask[not_completed]
        if self.cache is not None:
            self.cache = [
                layer_cache.filter_batch_items(not_completed)
                for layer_cache in self.cache
            ]
        self.positions = self.positions[not_completed]
        self.seq_ids = self.seq_ids[not_completed]
