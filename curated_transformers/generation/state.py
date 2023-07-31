from typing import Generic, List, Optional, Tuple

import torch
from torch import Tensor

from ..layers.attention import AttentionMask
from ..models.output import CacheT
from .stop_conditions import StopCondition


class GeneratorState(Generic[CacheT]):
    """
    Stores the state of the generation process and tracks
    the sequences being generated.
    """

    attention_mask: AttentionMask
    cache: Optional[List[CacheT]]
    positions: Tensor
    seq_ids: Tensor
    prompt_ids: Tensor
    generated_ids: Tensor

    def __init__(
        self,
        *,
        attention_mask: AttentionMask,
        cache: Optional[List[CacheT]],
        prompt_ids: Tensor,
    ) -> None:
        """
        Construct a generator state.

        :param attention_mask:
            Attention mask for the prompts.
        :param cache:
            Transformer model cache.
        :param prompt_ids:
            Batch of prompts.

            *Shape:* ``(batch_size, seq_len)``
        """
        device = prompt_ids.device
        assert (
            attention_mask.device == device
        ), f"Attention mask device '{attention_mask.device}' is not same as prompt ids device '{prompt_ids.device}'"
        self.attention_mask = attention_mask
        self.positions = attention_mask.bool_mask.int().cumsum(-1) - 1
        self.cache = cache
        self.seq_ids = torch.arange(0, self.attention_mask.shape[0], device=device)
        self.prompt_ids = prompt_ids
        self.generated_ids = torch.zeros(
            (prompt_ids.size(0), 0), dtype=prompt_ids.dtype, device=device
        )

    @property
    def is_finished(self):
        """
        Whether all sequences have finished generating.

        :returns:
            ``True`` iff all sequences have finished generating.
        """
        return len(self.seq_ids) == 0

    @property
    def last_step_ids(self) -> Tensor:
        """
        Identifiers generated in the last step.

        :returns:
            Generated identifiers. Prompt identifiers are returned
            when the generator has not stepped yet.
        """
        if not self.generated_ids.size(1):
            return self.prompt_ids
        else:
            return self.generated_ids[:, -1:]

    def step(
        self,
        *,
        cache: List[CacheT],
        predicted_ids: Tensor,
        stop_condition: StopCondition,
    ) -> Tuple[Tensor, Tensor]:
        """
        Step the generation state.

        :param cache:
            Model cache from the last model call.
        :param generated_ids:
            Tensor containing generated IDs.

            *Shape:* ``(batch_size, 1)``
        :param stop_condition:
            Generation stop condition.
        :returns:
            Sequence identifiers and piece IDs.

            *Shape:* ``(batch_size), (batch_size, 1)``
        """
        # We update the state before removing completed sequences, so that
        # stopping conditions get a consistent view.
        self.cache = cache
        self.generated_ids = torch.concat([self.generated_ids, predicted_ids], 1)
        self.attention_mask = self.attention_mask.extend_length(
            count=1, fill_value=True
        )
        self.positions = self.positions.max(-1, keepdim=True).values + 1

        # Determine which sequences are done generating and remove them.
        completed_exclude = torch.zeros_like(predicted_ids, dtype=torch.bool)
        completed_include = torch.zeros_like(predicted_ids, dtype=torch.bool)
        stop_condition.update_completed(
            state=self,
            completed_exclude=completed_exclude,
            completed_include=completed_include,
        )

        # Prepare sequences and identifiers that the generator should yield.
        to_yield = completed_exclude.logical_not().view(-1)
        seq_ids = self.seq_ids[to_yield.view(-1)]
        last_step_ids = self.last_step_ids[to_yield]

        self._remove_completed((completed_exclude ^ completed_include).view(-1))

        return seq_ids, last_step_ids

    def _remove_completed(self, completed: Tensor):
        """
        Remove completed sequences.

        :param completed:
            Tensor indicating for the active sequences whether they are completed.

        :meta private:
        """
        not_completed = completed.logical_not()
        self.generated_ids = self.generated_ids[not_completed]
        self.attention_mask = self.attention_mask.filter_batch_items(not_completed)
        if self.cache is not None:
            self.cache = [
                layer_cache.filter_batch_items(not_completed)
                for layer_cache in self.cache
            ]
        self.prompt_ids = self.prompt_ids[not_completed]
        self.positions = self.positions[not_completed]
        self.seq_ids = self.seq_ids[not_completed]
