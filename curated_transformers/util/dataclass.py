from dataclasses import fields
from typing import Any, OrderedDict, Tuple

from torch import Tensor


class DataclassAsDict(OrderedDict[str, Tensor]):
    """
    Dataclasses that derive from this struct are also a dictionary.

    Since this class should only be used for dataclasses that are
    ``torch.jit.trace``-able, only ``Tensor`` fields are supported.

    Only dataclass fields and keys corresponding to those fields
    can be changed. Fields and keys cannot be removed.
    """

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            # if value is None:
            #    continue
            # if isinstance(value, list):
            #    value = tuple(value)
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"DataclassAsDict only supports Tensor members, field '{field.name}' has type '{field.type.__name__}'"
                )

            super().__setitem__(field.name, value)

    def __delitem__(self, key: str):
        raise NotImplementedError()

    def __delattr__(self, name: str):
        raise NotImplementedError()

    def __setattr__(self, name: str, value: Any) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Field '{name}' cannot be set to non-Tensor type '{type(value).__name__}'"
            )

        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key: str, value: Tensor) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Key cannot be set to non-str type '{type(key).__name__}'")

        if not isinstance(value, Tensor):
            raise TypeError(
                f"Field '{key}' cannot be set to non-Tensor type '{type(value).__name__}'"
            )

        super().__setattr__(key, value)
        super().__setitem__(key, value)


class DataclassAsTuple:
    """
    Dataclasses that derive from this struct can be converted to a tuple.

    Since this class should only be used for dataclasses that are
    ``torch.jit.trace``-able, only the following types of fields
    are supported:

    * ``Tensor``
    * ``List[Tensor]``
    * ``List[DataclassAsDict]``
    * ``Optional`` of any of the above.

    Fields that have the value ``None`` are skipped.
    """

    def astuple(self) -> Tuple:
        to_tuples = []
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            elif isinstance(value, Tensor):
                pass
            elif isinstance(value, list):
                if all(isinstance(item, Tensor) for item in value):
                    value = tuple(value)
                elif all(isinstance(item, DataclassAsDict) for item in value):
                    value = tuple(value)
                else:
                    type_names = ", ".join(
                        sorted({type(item).__name__ for item in value})
                    )
                    raise TypeError(
                        f"List must be List[Tensor] or List[DataclassAsDict], found types: {type_names}"
                    )
            else:
                raise TypeError(
                    f"Field '{field.name}' has unsupported type '{type(value).__name__}'"
                )

            to_tuples.append(value)

        return tuple(to_tuples)
