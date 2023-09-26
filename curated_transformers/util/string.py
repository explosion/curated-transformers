import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class StringTransform(ABC):
    """
    Base class for reversible string transformations.
    """

    def __init__(self, reversible: bool = True):
        super().__init__()
        self._reversible = reversible

    @abstractmethod
    def _apply(self, string: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def _revert(self, string: str) -> str:
        raise NotImplementedError

    def apply(self, string: str) -> str:
        """
        Applies the transformation to the given string.

        :param string:
            String to transform.
        :returns:
            Transformed string.
        """
        return self._apply(string)

    def revert(self, string: str) -> str:
        """
        Reverts the previously applied transformation of the given string.

        :param string:
            Previously transformed string.
        :returns:
            Reverted string.
        """
        if self._reversible:
            return self._revert(string)
        else:
            return string


class StringSubRegEx(StringTransform):
    """
    Substitute a substring with another string using
    regular expressions.
    """

    def __init__(self, forward: Tuple[str, str], backward: Optional[Tuple[str, str]]):
        """
        Construct a reversible substitution.

        :param forward:
            Tuple where the first string is a RegEx pattern
            and the second the replacement.

            This operation is performed when the :meth:`.apply`
            method is invoked.
        :param backward:
            Optional tuple where the first string is a RegEx pattern
            and the second the replacement.

            This operation is performed when the :meth:`.revert`
            method is invoked. If ``None``, it is a no-op.
        """
        super().__init__(backward is not None)

        self.forward = forward
        self.backward = backward

    def _apply(self, string: str) -> str:
        return re.sub(self.forward[0], self.forward[1], string)

    def _revert(self, string: str) -> str:
        if self.backward is None:
            raise ValueError("Attempting to revert an irreversible string transform")
        return re.sub(self.backward[0], self.backward[1], string)


class StringSubInvertible(StringSubRegEx):
    """
    A substitution whose backward transformation can be
    automatically derived from the forward transformation.
    """

    def __init__(self, forward: Tuple[str, str]):
        """
        Construct a reversible (and invertible) substitution.

        :param forward:
            Tuple where the first string is string to match
            and the second the replacement, neither of which
            can contain RegEx meta-characters.
        """
        super().__init__(
            forward=(f"{re.escape(forward[0])}", forward[1]),
            backward=(f"{re.escape(forward[1])}", forward[0]),
        )


class StringReplace(StringTransform):
    """
    Replaces an entire string with another.
    """

    def __init__(self, replacee: str, replacement: str, *, reversible: bool = True):
        """
        Construct a reversible replacement.

        :param replacee:
            The full string to be replaced.
        :param replacement:
            The replacement string.
        """
class StringRemovePrefix(StringSubRegEx):
    """
    Strips a prefix from a given string.
    """

    def __init__(self, prefix: str, *, reversible: bool = True):
        """
        Construct a reversible left strip.

        :param prefix:
            Prefix to be stripped.
        """
        super().__init__(
            forward=(f"^{re.escape(prefix)}", ""),
            backward=(r"^(.)", f"{prefix}\\1") if reversible else None,
        )
