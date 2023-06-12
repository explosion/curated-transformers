from abc import ABC, abstractmethod
from dataclasses import dataclass

from .logits import (
    CompoundLogitTransforms,
    LogitsTransform,
    TemperatureTransform,
    TopKTransform,
)


class GeneratorConfig(ABC):
    """Configuration of the generator."""

    @abstractmethod
    def logits_transform(self) -> LogitsTransform:
        """
        Get logit transform for the configuration.

        :returns:
            Logits transform. Usually multiple composed transforms.
        """
        ...


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
        Softmax temperature. For a temperature T:

        - T = 1: the distribution is not changed.
        - T < 1: the entropy of the distribution is decreased.
        - T > 1: the entropy of the distribution is increased.
    :param top_k:
        Sample from top-k highest-probability pieces. k < 1 disables top-k
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
