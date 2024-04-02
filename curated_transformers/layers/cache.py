from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Type, TypeVar, Union

import torch
from torch import Tensor

CacheProtocolSelf = TypeVar("CacheProtocolSelf", bound="CacheProtocol")


class CacheProtocol(Protocol):
    def filter_batch_items(self: CacheProtocolSelf, mask: Tensor) -> CacheProtocolSelf:
        """
        Filter batch sequences from the cache.

        Sequences for which the mask is ``True`` are retained.

        :param mask:
            Mask of batch items to retain.

            *Shape:* ``(batch_size,)``
        :returns:
            Filtered items.
        """
        ...


@dataclass
class KeyValueCache:
    """
    Cache type for layers that cache keys and values.

    :param key:
        Key.
    :param value:
        Value.
    """

    key: Tensor
    value: Tensor

    def filter_batch_items(self, mask: Tensor) -> "KeyValueCache":
        if mask.ndim != 1:
            raise ValueError(
                f"Cache mask must be a 1D tensor, has {mask.ndim} dimensions."
            )
        if mask.size(0) != self.key.size(0):
            raise ValueError(
                f"Cache mask size ({mask.size(0)}) must match cache batch size ({self.key.size(0)})."
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"Cache mask dtype must be bool, was: {mask.dtype}.")

        return KeyValueCache(key=self.key[mask], value=self.value[mask])
