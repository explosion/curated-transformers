from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Type, TypeVar, Union

import torch
from torch import Tensor

from curated_transformers.util.dataclass import DataclassAsDict, DataclassAsTuple

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
class KeyValueCache(DataclassAsDict):
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

        return KeyValueCache(self.key[mask], self.value[mask])

    @classmethod
    def jit_rewrap(
        cls: Type["KeyValueCache"],
        key_value_cache: Optional[Union["KeyValueCache", Dict[str, Tensor]]],
    ) -> Optional["KeyValueCache"]:
        """
        Rewrap TorchScript dictionary conversion of a key-value cache.

        :param key_value_cache:
            The key-value cache or its dictionary representation. If the
            value is a ``KeyValueCache`` or ``None``, it will be
            returned as-is.
        :returns:
            The key-value cache.
        """
        if key_value_cache is None or isinstance(key_value_cache, KeyValueCache):
            return key_value_cache

        key = key_value_cache.get("key")
        if key is None:
            raise ValueError(
                "Key-value cache is not of the `KeyValueCache` type, nor a dict with 'key'."
            )

        value = key_value_cache.get("value")
        if value is None:
            raise ValueError(
                "Key-value cache is not of the `KeyValueCache`` type, nor a dict with 'value'."
            )

        return cls(key=key, value=value)
