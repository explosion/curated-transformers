from typing import Type, TypeVar, Union


class Default:
    """
    Marker type to be used as a default value for
    :py:class:`~curated_transformers.semver.FutureMandatory`.
    """

    def __init__(self):
        raise TypeError(
            f"{type(self).__name__} is a marker class and cannot be constructed."
        )


T = TypeVar("T")

#: ``FutureMandatory[T]`` is either an instance of ``T`` or the type
#: :py:class:`~curated_transformers.semver.Default`. It used as the
#: type for arguments that will become mandatory in the next major
#: version.
FutureMandatory = Union[T, Type[Default]]
