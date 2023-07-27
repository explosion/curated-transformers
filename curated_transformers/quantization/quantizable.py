from abc import ABC, abstractmethod
from typing import Set


class Quantizable(ABC):
    """
    Mixin class for models that are quantizable.

    A module using this mixin provides the necessary configuration
    and parameter information to quantize it on-the-fly during the
    module loading phase.
    """

    @classmethod
    @abstractmethod
    def modules_to_not_quantize(cls) -> Set[str]:
        """
        Return a set of prefixes that specify which
        modules are to be ignored during quantization.

        :returns:
            Set of module prefixes.

            If empty, all submodules will be quantized.
        """
        raise NotImplementedError
