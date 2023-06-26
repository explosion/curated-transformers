from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from torch import Tensor

if TYPE_CHECKING:
    from .state import GeneratorState


class StopCondition(ABC):
    """
    Base class for generation stop conditions.
    """

    @abstractmethod
    def update_completed(
        self,
        *,
        state: "GeneratorState",
        completed_exclude: Tensor,
        completed_include: Tensor
    ):
        """
        Update completed sequences according to the stop condition.

        :param completed_exclude:
            Output tensor marking which sequences are completed and
            for which the last generated piece **should not** be emitted.
        :param completed_include:
            Output tensor marking which sequences are completed and
            for which the last generated piece **should** be emitted.
        """
        ...


class CompoundStopCondition(List[StopCondition], StopCondition):
    """
    Sequentially apply multiple stop conditions.
    """

    def update_completed(
        self,
        *,
        state: "GeneratorState",
        completed_exclude: Tensor,
        completed_include: Tensor
    ):
        for condition in self:
            condition.update_completed(
                state=state,
                completed_exclude=completed_exclude,
                completed_include=completed_include,
            )


class EndOfSequenceCondition(StopCondition):
    """
    Stop when the end-of-sequence piece is predicted.
    """

    def __init__(self, eos_id: int):
        """
        Construct the stop condition.

        :param eos_id:
            End-of-sequence identifier that marks the end of a generated
            sequence.
        """
        self.eos_id = eos_id

    def update_completed(
        self,
        *,
        state: "GeneratorState",
        completed_exclude: Tensor,
        completed_include: Tensor
    ):
        completed_exclude ^= state.last_step_ids == self.eos_id


class MaxGeneratedPiecesCondition(StopCondition):
    """
    Stop after generating a maximum number of pieces.
    """

    def __init__(self, max_generated_pieces: int):
        """
        Construct the stop condition.

        :param max_generated_pieces:
            The maximum number of generated pieces. This condition is a noop
            for values less than 1.
        """
        super().__init__()
        self.max_generated_pieces = max_generated_pieces

    def update_completed(
        self,
        *,
        state: "GeneratorState",
        completed_exclude: Tensor,
        completed_include: Tensor
    ):
        if self.max_generated_pieces < 1:
            return False

        if state.generated_ids.size(-1) >= self.max_generated_pieces:
            completed_include ^= True
