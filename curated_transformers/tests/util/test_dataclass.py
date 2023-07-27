from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pytest
import torch
from torch import Tensor

from curated_transformers.util.dataclass import DataclassAsDict, DataclassAsTuple


@dataclass
class AsDict(DataclassAsDict):
    foo: Tensor
    bar: Tensor


@dataclass
class AsTuple(DataclassAsTuple):
    foo: Tensor
    bar: List[Tensor]
    baz: Optional[Tensor]
    quux: List[Tuple[Any]]


@dataclass
class InvalidAsTuple(DataclassAsTuple):
    bar: int


@dataclass
class InvalidAsDict(DataclassAsDict):
    foo: Tensor
    bar: int


def test_as_dict():
    d = AsDict(foo=torch.zeros(1, 1), bar=torch.full((2, 2), 42))
    assert "foo" in d
    assert "bar" in d
    assert d["foo"] is d.foo
    assert d["bar"] is d.bar

    d.bar = torch.full((3, 3), 80)
    assert d["bar"] is d.bar

    d["foo"] = torch.ones((4, 4))
    assert d["foo"] is d.foo


def test_as_dict_non_tensor_member_rejected():
    with pytest.raises(TypeError, match=r"bar.*int"):
        InvalidAsDict(foo=torch.zeros(1, 1), bar=42)


def test_as_dict_invalid_operations():
    d = AsDict(foo=torch.zeros(1, 1), bar=torch.full((2, 2), 42))
    with pytest.raises(TypeError, match=r"foo.*str"):
        d.foo = "Glove80"
    with pytest.raises(TypeError, match=r"foo.*str"):
        d["foo"] = "Glove80"
    with pytest.raises(TypeError, match=r"non-`str`.*int"):
        d[83] = "Glove80"
    with pytest.raises(NotImplementedError):
        del d["foo"]
    with pytest.raises(NotImplementedError):
        delattr(d, "foo")


def test_as_tuple():
    t = AsTuple(
        torch.full((2, 2), 42),
        [torch.ones(3, 3), torch.zeros(4, 4)],
        None,
        [(42, 80)],
    )

    assert t[0] is t.foo

    assert len(t[1]) == 2
    assert t[1][0] is t.bar[0]
    assert t[1][1] is t.bar[1]

    assert len(t[2]) == 1
    assert t[2][0] is t.quux[0]


def test_as_tuple_incorrect():
    with pytest.raises(TypeError, match=r"Tensor, str"):
        AsTuple(
            torch.full((2, 2), 42),
            [torch.ones(3, 3), "Glove80"],
            None,
            [AsDict(torch.full((5, 5), 80), torch.full((6, 6), -1))],
        )

    with pytest.raises(TypeError, match=r"unsupported.*int"):
        InvalidAsTuple(42)


def test_as_tuple_is_immutable():
    t = AsTuple(
        torch.full((2, 2), 42),
        [torch.ones(3, 3), torch.zeros(4, 4)],
        None,
        [(42, 80)],
    )

    with pytest.raises(TypeError, match=r"does not support attribute assignment"):
        t.foo = torch.zeros(5, 5)

    with pytest.raises(TypeError, match=r"does not support attribute deletion"):
        del t.foo
