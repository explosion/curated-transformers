from typing import Any, Generic, Iterator, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical

from ..layers.attention import AttentionMask
from ..models.module import CausalLMModule
from ..models.output import CacheT, CausalLMOutputWithCache
from .config import GeneratorConfig, GreedyGeneratorConfig, SampleGeneratorConfig
from .logits import LogitsTransform
from .state import GeneratorState


class Generator(Generic[CacheT]):
    """
    Generator base class for causal language models.
    """

    model: CausalLMModule[Any, CacheT]

    def __init__(self, model: CausalLMModule[Any, CacheT]):
        """
        Construct a generator.

        :param model:
            The causal language model to generate with.
        """
        self.model = model

    def __call__(
        self,
        *,
        attention_mask: AttentionMask,
        ids: Tensor,
        config: GeneratorConfig,
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Alias for :meth:`.generate`.
        """
        return self.generate(
            attention_mask=attention_mask,
            ids=ids,
            config=config,
        )

    def generate(
        self,
        *,
        attention_mask: AttentionMask,
        ids: Tensor,
        config: GeneratorConfig,
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Generate text, starting from the given piece identifiers.

        The generator returns an iterator over tuples. Each tuple contains:
         1. A tensor with sequence identifiers.
         2. A tensor with the next piece identifiers.

        The sequence identifiers are numbered ``0..batch`` and are
        necessary because some sequences may finish generation earliers than
        others. The sequence identifiers allow the caller to map the generated
        pieces back to the original input sequences.

        :param ids:
            Batch of piece identifiers to start generating from.

            *Shape:* ``(batch_size, seq_len)``
        :param attention_mask:
            Attention mask that masks out pieces that should not be attended to.
        :param config:
            Generator configuraton.
        :returns:
            An iterator over tuples. Each tuple contains a tensor with the
            sequence identifiers and a tensor with the next piece identier.

            *Shape:* ``(batch_unfinished,)``
        """
        self.model.eval()

        logits_transform = config.logits_transform()
        stop_condition = config.stop_condition()
        if isinstance(config, GreedyGeneratorConfig):
            generation_step = self._decode_greedy
        elif isinstance(config, SampleGeneratorConfig):
            generation_step = self._decode_sample
        else:
            raise ValueError(
                f"Unknown generator configuration: {type(config).__name__}"
            )

        cache: Optional[List[CacheT]] = None
        state = GeneratorState(
            attention_mask=attention_mask, cache=cache, prompt_ids=ids
        )

        while True:
            with torch.no_grad():
                output = self.model(
                    state.last_step_ids,
                    attention_mask=state.attention_mask,
                    cache=state.cache,
                    store_cache=True,
                    positions=state.positions,
                )

            seq_ids, last_step_ids = state.step(
                cache=output.cache,
                predicted_ids=generation_step(logits_transform, output),
                stop_condition=stop_condition,
            )

            if seq_ids.size(0) > 0:
                yield seq_ids, last_step_ids

            if state.is_finished:
                return

    def _decode_greedy(
        self,
        logits_transform: LogitsTransform,
        output: CausalLMOutputWithCache[CacheT],
    ) -> Tensor:
        logits = logits_transform(output.logits[:, -1:, :], inplace=True)
        return logits.argmax(-1)

    def _decode_sample(
        self, logits_transform: LogitsTransform, output: CausalLMOutputWithCache[CacheT]
    ) -> Tensor:
        logits = logits_transform(output.logits[:, -1:, :], inplace=True)
        distribution = Categorical(logits=logits)
        return distribution.sample()
