from typing import Generic, Iterator, List, Optional, Tuple
import torch
from torch import Tensor

from ..models.attention import AttentionMask, CacheT
from ..models.module import CausalLMModule
from .state import GeneratorState


class Generator(Generic[CacheT]):
    """Generator for causal language models."""

    model: CausalLMModule[CacheT]

    def __init__(self, model: CausalLMModule[CacheT]):
        """
        Construct a generator.

        :param model:
            The causal language model to generate with.
        """
        self.model = model

    def __call__(
        self, *, attention_mask: Tensor, ids: Tensor, eos_id: int
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        See the :meth:`.generate` method.
        """
        return self.generate(attention_mask=attention_mask, ids=ids, eos_id=eos_id)

    def generate(
        self, *, attention_mask: Tensor, ids: Tensor, eos_id: int
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Generate text, starting from the given piece identifiers.

        The generator returns an iterator over tuples. Each tuple contains (1) a
        tensor with sequence identifiers; (2) a tensor with the next piece
        identifiers. The sequence identifiers are numbered 0..batch and are
        necessary because some sequences may finish generation earliers than
        others. The sequence identifiers allow the caller to map the generated
        pieces back to the original input sequences.

        :param ids:
            Batch of piece identifiers to start generating from.
            **Shape:** (batch, seq_len)
        :param attention_mask:
            Attention mask that masks out pieces that should not be attended to.
            **Shape:** (batch, seq_len)
        :param eos_id:
            Piece identifier that signals the end of the generated sequence.
        :returns:
            An iterator over tuples. Each tuple contains a tensor with the
            sequence identifiers and a tensor with the next piece identier.
            **Shape:** (batch_unfinished,)
        """
        self.model.eval()

        cache: Optional[List[CacheT]] = None
        state = GeneratorState(attention_mask=attention_mask, cache=cache)

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
            state.step(cache=output.cache, completed=completed)

            if state.is_finished:
                return

            yield state.seq_ids, ids
