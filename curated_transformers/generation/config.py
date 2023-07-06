from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .logits import (
    CompoundLogitTransforms,
    LogitsTransform,
    TemperatureTransform,
    TopKTransform,
)
from .stop_conditions import (
    CompoundStopCondition,
    EndOfSequenceCondition,
    MaxGeneratedPiecesCondition,
    StopCondition,
)


@dataclass
class GeneratorConfig(ABC):
    """
    Configuration of the generator.

    :param eos_id:
        End-of-sequence identifier that should end the generation of a sequence
        when predicted. When this value is set to `None`, it is the
        responsibility of the generator to set it.

    :param max_generated_pieces:
        The maximum number of generation steps. This condition is a noop
        for values less than 1. When this value is set to `None`, it is the
        responsibility of the generator to set it.
    """

    eos_id: Optional[int] = None
    max_generated_pieces: Optional[int] = None

    @abstractmethod
    def logits_transform(self) -> LogitsTransform:
        """
        Get logit transform for the configuration.

        :returns:
            Logits transform. Usually multiple composed transforms.
        """
        ...

    def stop_condition(self) -> StopCondition:
        """
        Get the stop condition for the configuration.

        :returns:
            Stop condition. Usually multiple composed conditions.
        """
        conditions = CompoundStopCondition()
        if self.eos_id is None:
            raise ValueError("End-of-sequence piece id is unset")
        else:
            conditions.append(EndOfSequenceCondition(self.eos_id))

        if self.max_generated_pieces is None:
            raise ValueError("Maximum number of generation steps is unset")
        else:
            conditions.append(
                MaxGeneratedPiecesCondition(
                    max_generated_pieces=self.max_generated_pieces
                )
            )

        return conditions


class GreedyGeneratorConfig(GeneratorConfig):
    """
    Configuration for greedy generation.

    Greedy generation always selects the highest-probability piece, leading
    to deterministic generation.
    """

    def logits_transform(self) -> LogitsTransform:
        return CompoundLogitTransforms([])


@dataclass
class SampleGeneratorConfig(GeneratorConfig):
    """
    Configuration for generation with sampling.

    Sampling-based generation samples pieces from probability distributions.
    Generation is non-deterministic as a result, but provides more varied
    output.

    :param temperature:
        Softmax temperature. For a temperature ``T``:

        - ``T = 1``: the distribution is not changed.
        - ``T < 1``: the entropy of the distribution is decreased.
        - ``T > 1``: the entropy of the distribution is increased.
    :param top_k:
        Sample from top-k highest-probability pieces. ``top_k < 1`` disables top-k
        filtering.
    """

    temperature: float = 1.0
    top_k: int = 0

    def logits_transform(self) -> LogitsTransform:
        return CompoundLogitTransforms(
            [
                TemperatureTransform(self.temperature),
                TopKTransform(self.top_k),
            ]
        )
