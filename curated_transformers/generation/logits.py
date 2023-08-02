from abc import ABC, abstractmethod
from collections import UserList
from typing import Iterable

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


class CompoundLogitsTransform(UserList, LogitsTransform):
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
            The value of k in top-k. The transform is a no-op for values
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


class TopPTransform(LogitsTransform):
    """
    Keep the smallest possible set of most probable vocab items, such that
    their cumulative probability is >= p. Sampling using the top-p transform
    is also known as nucleus sampling (`Holzman et al., 2019`_). The
    probability of the items that are masked out is redistributed across the
    top-p items.

    .. _Holzman et al., 2019: https://arxiv.org/abs/1904.09751
    """

    def __init__(self, p: float):
        """
        Construct a top-p logits transform.

        :param p:
            The value of p in top-p. The transform is a no-op for
            ``p = 1.0``.
        """
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"Top-p probability must be between 0 and 1 (inclusive), was: {p}"
            )
        self.p = p

    def _process_logits(self, logits: Tensor):
        if self.p == 1.0:
            return

        sorted_values, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_values.softmax(-1).cumsum(-1)

        retain_indices = cumulative_probs < self.p

        # We need to get the smallest set for which the cumulative
        # probability exceeds p, so we need an additional logit. We
        # never include one logit too many because of the use of
        # ``<`` rather than `<=` above.
        retain_indices = retain_indices.roll(1, dims=-1)
        retain_indices[:, 0] = True

        # Sorted values with masked-out logits.
        mask = torch.finfo(logits.dtype).min
        sorted_values = torch.where(retain_indices, sorted_values, mask)

        # Put back in the unsorted logits.
        logits.scatter_(-1, sorted_indices, sorted_values)


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


class VocabMaskTransform(LogitsTransform):
    """
    Set the probability of specific vocabulary pieces to zero.
    """

    def __init__(self, pieces_to_mask: Iterable[int]):
        """
        Construct a mask logits transform.

        :param pieces_to_mask:
            Identifers pertaining to the vocabulary pieces that
            need to be masked. An empty iterable results in a no-op.
        """
        super().__init__()

        self.pieces_to_mask = torch.tensor(
            list(pieces_to_mask), dtype=torch.long
        )  # `torch.int32` isn't supported in indexing operations prior in torch<2.0.0.

        if self.pieces_to_mask.dim() != 1:
            raise ValueError("Vocabulary piece mask must be 1D")
        elif (
            self.pieces_to_mask.size(dim=-1) != 0
            and int(self.pieces_to_mask.min(dim=-1).values) < 0
        ):
            raise ValueError("Vocabulary piece identifiers must be >= 0")

    def _process_logits(self, logits: Tensor):
        if self.pieces_to_mask.size(dim=-1) == 0:
            return

        try:
            mask = torch.finfo(logits.dtype).min
            logits[..., self.pieces_to_mask] = mask
        except IndexError:
            max_expected = logits.size(dim=-1)
            max_received = int(self.pieces_to_mask.max(dim=-1).values)
            if max_received >= max_expected:
                raise ValueError(
                    f"Vocabulary piece identifiers must be < {max_expected}, but got {max_received}"
                )
