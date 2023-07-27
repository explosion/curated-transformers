from dataclasses import fields
from typing import Any, Generator, OrderedDict

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
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"`DataclassAsDict` only supports `Tensor` members, but field '{field.name}' has type `{field.type.__name__}`"
                )

            super().__setitem__(field.name, value)

    def __delitem__(self, key: str):
        raise NotImplementedError()

    def __delattr__(self, name: str):
        raise NotImplementedError()

    def __setattr__(self, name: str, value: Any) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Field '{name}' cannot be set to non-Tensor type `{type(value).__name__}`"
            )

        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key: str, value: Tensor) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"Key cannot be set to non-`str` type `{type(key).__name__}`"
            )

        if not isinstance(value, Tensor):
            raise TypeError(
                f"Field '{key}' cannot be set to non-Tensor type `{type(value).__name__}`"
            )

        super().__setattr__(key, value)
        super().__setitem__(key, value)


class _InterceptGeneratorMeta(type):
    """
    Tuples can take a generator as their only constructor argument,
    dataclasses will see them them as a single argument. Intercept
    single generator argument, evaluate the generator, and pass
    the values as args.
    """

    def __new__(cls, name, bases, dict):
        # Add tuple as a base class. MyPy complains about having tuple
        # directly as a base class. See: https://github.com/python/mypy/issues/14818
        return super().__new__(cls, name, bases + (tuple,), dict)

    def __call__(cls, *args, **kwargs):
        # Convert generator argument to a tuple, so that we can pass
        # the values as regular arguments.
        if len(args) == 1 and kwargs == {} and isinstance(args[0], Generator):
            args = tuple(args[0])

        obj = super().__call__(*args, **kwargs)
        obj._is_frozen = True

        return obj


class DataclassAsTuple(metaclass=_InterceptGeneratorMeta):
    """
    Dataclasses that derive from this class are also a tuple.

    Since this class should only be used for dataclasses that are
    ``torch.jit.trace``-able, only the following types of fields
    are supported:

    * ``Tensor``
    * ``List[Tensor]``
    * ``List[Tuple[...]]``
    * ``Optional`` of any of the above.

    Fields that have the value ``None`` are skipped.
    """

    def __new__(cls, *args, **kwargs):
        values = []
        for idx, field in enumerate(fields(cls)):
            if idx < len(args):
                value = args[idx]
            elif field.name in kwargs:
                value = kwargs[field.name]
            else:
                # Field is not specified, consider it optional. If it is
                # a mandatory field, the dataclass machinery will complain
                # later.
                continue

            if value is None:
                continue

            values.append(DataclassAsTuple._convert_value(value))

        return tuple.__new__(cls, values)

    @staticmethod
    def _convert_value(value):
        if isinstance(value, Tensor):
            return value
        elif isinstance(value, list):
            if all(isinstance(item, Tensor) for item in value):
                return tuple(value)
            elif all(isinstance(item, tuple) for item in value):
                return tuple(value)
            else:
                type_names = ", ".join(sorted({type(item).__name__ for item in value}))
                raise TypeError(
                    f"List must be `List[Tensor]` or `List[Tuple[...]]`, found types: {type_names}"
                )
        else:
            raise TypeError(f"Field has unsupported type `{type(value).__name__}`")

    def __delattr__(self, name: str):
        if getattr(self, "_is_frozen", False):
            raise TypeError("`AsTuple` object does not support attribute deletion")
        super().__delattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_is_frozen", False):
            raise TypeError("`AsTuple` object does not support attribute assignment")
        super().__setattr__(name, value)
