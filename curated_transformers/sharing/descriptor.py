from dataclasses import dataclass
from enum import Enum, IntEnum

from torch.nn import Identity, Module, Parameter


class SharedDataType(Enum):
    """
    Type of shared data.
    """

    MODULE = 0
    PARAMETER = 1


@dataclass(frozen=True)
class SharedDataDescriptor:
    """
    Describes data that is shared between modules.

    :param source:
        Fully-qualified name of the parameter or module that will be shared.
    :param target:
        Fully-qualified name of the parameter or module that will be "replaced"
        by the source.
    :param type:
        Type of data being shared.
    """

    source: str
    target: str
    type: SharedDataType
