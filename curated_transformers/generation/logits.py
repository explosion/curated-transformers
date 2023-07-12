from abc import ABC, abstractmethod
from collections import UserList

import torch
from torch import Tensor


class LogitsTransform(ABC):
    """
    A logits transform changes the logits of a softmax distribution in some
    way. For instance, :class:`TemperatureTransform` changes the temperature
    of the softmax distribution.
    """

    def __call__(self, logits: Tensor, inplace: bool = False) -> Tensor:
        """
        Transform the given logits.

        :param logits:
            The logits array.

            *Shape:* ``(..., n_class)``
        :param inplace:
            Transform logits in-place.
        :returns:
            The transformed logits. This will be the same tensor object as
            the ``logits`` argument when ``inplace`` is set.

            *Shape:* ``(..., n_class)``
        """
        if not inplace:
            logits = logits.clone()
        self._process_logits(logits)
        return logits

    @abstractmethod
    def _process_logits(self, logits: Tensor):
        """
        Transform logits in-place.

        :param logits:
            The logits array.

            *Shape:* ``(..., n_class)``
        """
        ...


class CompoundLogitTransforms(UserList, LogitsTransform):
    """
    Sequentially apply multiple logit transforms.
    """

    def _process_logits(self, logits: Tensor):
        for transform in self:
            logits = transform(logits, inplace=True)


class TopKTransform(LogitsTransform):
    """
    Set the probability of non-top-k classes to zero. The probability of
    the classes that are zeroed out is redistributed across the top-k
    classes.
    """

    def __init__(self, k: int):
        """
        Construct a top-k logits transform.

        :param k:
            The value of k in top-k. The transform is a noop for values
            less than 1.
        """
        super().__init__()
        self.k = k

    def _process_logits(self, logits: Tensor):
        if self.k < 1:
            return

        # Nothing to do if we have k or fewer classes.
        if logits.size(-1) <= self.k:
            return

        mask = torch.finfo(logits.dtype).min
        cutoff = torch.topk(logits, self.k, dim=-1).values[..., -1:]
        logits[logits < cutoff] = mask


class TemperatureTransform(LogitsTransform):
    """
    Apply temperature to the softmax distribution. Given the temperature ``T``
    and logits ``z(y|x)``:

    .. math::
        p(y|x) = softmax(z(y|x)/T)

    For a temperature ``T``:

    - ``T = 1``: the distribution is not changed.
    - ``T < 1``: the entropy of the distribution is decreased.
    - ``T > 1``: the entropy of the distribution is increased.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Create a temperature transform with a given temperature.

        :param temperature:
            The temperature. Must be a non-zero positive value.
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(
                f"The temperature must be a non-zero positive value, was: {temperature}"
            )
        self.temperature = temperature

    def _process_logits(self, logits: Tensor):
        if self.temperature == 1.0:
            return

        logits /= self.temperature
