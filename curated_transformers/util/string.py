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


class StringTransformations:
    """
    Provides factory methods for different string transformations.
    """

    @staticmethod
    def regex_sub(
        forward: Tuple[str, str], backward: Optional[Tuple[str, str]]
    ) -> StringTransform:
        """
        Factory method to construct a string substitution transform
        using regular expressions.

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
        return StringSubRegEx(forward, backward)

    @staticmethod
    def sub(
        substring: str, replacement: str, *, reversible: bool = True
    ) -> StringTransform:
        """
        Factory method to construct a string substitution transform.

        :param substring:
            The substring to be replaced.
        :param replacement:
            The replacement string.
        :param reversible:
            If the reverse transformation is to
            be performed.
        """
        return StringSub(substring, replacement, reversible=reversible)

    @staticmethod
    def replace(
        replacee: str, replacement: str, *, reversible: bool = True
    ) -> StringTransform:
        """
        Factory method to construct a full string replacement transform.

        :param replacee:
            The full string to be replaced.
        :param replacement:
            The replacement string.
        :param reversible:
            If the reverse transformation is to
            be performed.
        """
        return StringReplace(replacee, replacement, reversible=reversible)

    @staticmethod
    def remove_prefix(prefix: str, *, reversible: bool = True) -> StringTransform:
        """
        Factory method to construct a string prefix removal transform.

        :param prefix:
            Prefix to be removed.
        :param reversible:
            If the reverse transformation is to
            be performed.
        """
        return StringRemovePrefix(prefix, reversible=reversible)


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


class StringSub(StringTransform):
    """
    Substitute a substring with another string.
    """

    def __init__(self, substring: str, replacement: str, *, reversible: bool = True):
        """
        Construct a reversible substitution.

        :param substring:
            The substring to be replaced.
        :param replacement:
            The replacement string.
        :param reversible:
            If the reverse transformation is to
            be performed.
        """
        super().__init__(reversible)
        self.substring = substring
        self.replacement = replacement

    def _apply(self, string: str) -> str:
        return string.replace(self.substring, self.replacement)

    def _revert(self, string: str) -> str:
        return string.replace(self.replacement, self.substring)


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
        :param reversible:
            If the reverse transformation is to
            be performed.
        """
        super().__init__(reversible)
        self.replacee = replacee
        self.replacement = replacement

    def _apply(self, string: str) -> str:
        if string == self.replacee:
            return self.replacement
        else:
            return string

    def _revert(self, string: str) -> str:
        if string == self.replacement:
            return self.replacee
        else:
            return string


class StringRemovePrefix(StringTransform):
    """
    Strips a prefix from a given string.
    """

    def __init__(self, prefix: str, *, reversible: bool = True):
        """
        Construct a reversible left strip.

        :param prefix:
            Prefix to be removed.
        :param reversible:
            If the reverse transformation is to
            be performed.
        """

        super().__init__(reversible)
        self.prefix = prefix

    def _apply(self, string: str) -> str:
        # TODO: Should be replaced with `removeprefix` once
        # Python 3.9 is the minimum requirement.
        return re.sub(f"^{re.escape(self.prefix)}", "", string)

    def _revert(self, string: str) -> str:
        return f"{self.prefix}{string}"
