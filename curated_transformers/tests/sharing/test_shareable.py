from typing import Set

import pytest
import torch
from torch.nn import Embedding, Linear, Module, Parameter

from curated_transformers.sharing import Shareable, SharedDataDescriptor, SharedDataType
from curated_transformers.sharing.logic.module import SharedModulePlaceholder
from curated_transformers.sharing.logic.parameter import SharedParameterPlaceholder


class _InnerModel(Module):
    def __init__(self, *args, device, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.inner_linear_1 = Linear(5, 10, device=device)
        self.inner_linear_2 = Linear(11, 17, device=device)
        self.inner_embedding_1 = Embedding(100, 10, device=device, dtype=torch.bfloat16)
        self.inner_embedding_2 = Embedding(100, 10, device=device, dtype=torch.float32)


class _MockModel(Module, Shareable):
    def __init__(self, *args, device=None, func=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Shareable.__init__(self)

        self.inner_model_1 = _InnerModel(device=device)
        self.inner_model_2 = _InnerModel(device=device)
        self.outer_linear = Linear(10, 10, bias=False, device=device)
        self.shared_data_callable = func

    def shared_data(self) -> Set[SharedDataDescriptor]:
        assert self.shared_data_callable is not None
        return self.shared_data_callable()


def _default_shared_data():
    return {
        SharedDataDescriptor(
            source="inner_model_2.inner_linear_1",
            target="inner_model_1.inner_linear_1",
            type=SharedDataType.MODULE,
        ),
        SharedDataDescriptor(
            source="inner_model_1.inner_linear_2.weight",
            target="inner_model_2.inner_linear_2.weight",
            type=SharedDataType.PARAMETER,
        ),
    }


def test_shareable_tying_untying_cloning():
    model = _MockModel(device="meta", func=_default_shared_data)

    assert type(model.inner_model_1.inner_linear_1) == Linear
    assert type(model.inner_model_1.inner_linear_2.weight) == Parameter

    model.initialize_shared_data()

    assert type(model.inner_model_1.inner_linear_1) == SharedModulePlaceholder
    assert model.inner_model_1.inner_linear_1.wrapped_source_type == Linear

    assert type(model.inner_model_2.inner_linear_2.weight) == SharedParameterPlaceholder
    assert model.inner_model_2.inner_linear_2.weight.shape == torch.Size([0])
    assert model.inner_model_2.inner_linear_2.weight.wrapped_shape == torch.Size(
        [17, 11]
    )
    assert (
        model.inner_model_2.inner_linear_2.weight.dtype
        == model.inner_model_2.inner_linear_2.weight.wrapped_dtype
    )

    model.tie_shared_data()

    assert model.tied
    assert id(model.inner_model_1.inner_linear_1) == id(
        model.inner_model_2.inner_linear_1
    )
    assert id(model.inner_model_1.inner_linear_2.weight) == id(
        model.inner_model_2.inner_linear_2.weight
    )

    model.untie_shared_data()
    assert not model.tied
    assert id(model.inner_model_1.inner_linear_1) != id(
        model.inner_model_2.inner_linear_1
    )
    assert id(model.inner_model_1.inner_linear_2.weight) != id(
        model.inner_model_2.inner_linear_2.weight
    )
    assert type(model.inner_model_1.inner_linear_1) == SharedModulePlaceholder
    assert type(model.inner_model_2.inner_linear_2.weight) == SharedParameterPlaceholder

    model.clone_shared_data()
    assert id(model.inner_model_1.inner_linear_1) != id(
        model.inner_model_2.inner_linear_1
    )
    assert id(model.inner_model_1.inner_linear_2.weight) != id(
        model.inner_model_2.inner_linear_2.weight
    )
    assert type(model.inner_model_1.inner_linear_1) == Linear
    assert type(model.inner_model_2.inner_linear_2.weight) == Parameter
    assert model.inner_model_2.inner_linear_2.weight.shape == torch.Size([17, 11])


def test_shareable_invalid_device():
    with pytest.raises(ValueError, match="non-meta shared"):
        model = _MockModel(device="cpu", func=_default_shared_data)
        model.initialize_shared_data()


def test_shareable_overlapping_shares():
    def shared_data():
        return {
            SharedDataDescriptor(
                "inner_model_1.inner_linear_1",
                "inner_model_2.inner_linear_1",
                SharedDataType.MODULE,
            ),
            SharedDataDescriptor(
                "inner_model_1.inner_linear_1.weight",
                "inner_model_2.inner_linear_1.weight",
                SharedDataType.PARAMETER,
            ),
        }

    with pytest.raises(ValueError, match="overlaps with shared module"):
        model = _MockModel(device="meta", func=shared_data)
        model.initialize_shared_data()


def test_shareable_initialized():
    model = _MockModel(device="meta", func=_default_shared_data)
    with pytest.raises(ValueError, match="not been initialized"):
        model.tie_shared_data()
    with pytest.raises(ValueError, match="not been initialized"):
        model.untie_shared_data()
    with pytest.raises(ValueError, match="not been initialized"):
        model.clone_shared_data()
    with pytest.raises(ValueError, match="already been initialized"):
        model.initialize_shared_data()
        model.initialize_shared_data()


def test_shareable_not_tied_or_untied():
    model = _MockModel(device="meta", func=_default_shared_data)
    model.initialize_shared_data()

    with pytest.raises(ValueError, match="Share the data first"):
        model.untie_shared_data()
    model.tie_shared_data()
    with pytest.raises(ValueError, match="Unshare the data first"):
        model.tie_shared_data()


def test_shareable_double_placeholder():
    def shared_data_param():
        return {
            SharedDataDescriptor(
                "inner_model_1.inner_linear_1.weight",
                "inner_model_2.inner_linear_1.weight",
                SharedDataType.PARAMETER,
            ),
            SharedDataDescriptor(
                "inner_model_2.inner_linear_2.weight",
                "inner_model_2.inner_linear_1.weight",
                SharedDataType.PARAMETER,
            ),
        }

    def shared_data_module():
        return {
            SharedDataDescriptor(
                "inner_model_1.inner_linear_1",
                "inner_model_2.inner_linear_1",
                SharedDataType.MODULE,
            ),
            SharedDataDescriptor(
                "inner_model_2.inner_linear_2",
                "inner_model_2.inner_linear_1",
                SharedDataType.MODULE,
            ),
        }

    with pytest.raises(ValueError, match="Shared parameter must be of type"):
        model = _MockModel(device="meta", func=shared_data_param)
        model.initialize_shared_data()

    with pytest.raises(
        ValueError, match="placeholder cannot target another placeholder"
    ):
        model = _MockModel(device="meta", func=shared_data_module)
        model.initialize_shared_data()


def test_shareable_mismatched_module_type():
    def shared_data_module():
        return {
            SharedDataDescriptor(
                "inner_model_1.inner_linear_1",
                "inner_model_2.inner_embedding_1",
                SharedDataType.MODULE,
            ),
        }

    with pytest.raises(ValueError, match="expected a source of type"):
        model = _MockModel(device="meta", func=shared_data_module)
        model.initialize_shared_data()
        model.tie_shared_data()


def test_shareable_mismatched_param_shape_type():
    def shared_data_dtype():
        return {
            SharedDataDescriptor(
                "inner_model_1.inner_embedding_1.weight",
                "inner_model_2.inner_embedding_2.weight",
                SharedDataType.PARAMETER,
            ),
        }

    with pytest.raises(ValueError, match="Mismatching dtypes"):
        model = _MockModel(device="meta", func=shared_data_dtype)
        model.initialize_shared_data()
        model.tie_shared_data()

    def shared_data_shape():
        return {
            SharedDataDescriptor(
                "inner_model_1.inner_linear_1.weight",
                "inner_model_2.inner_linear_2.weight",
                SharedDataType.PARAMETER,
            ),
        }

    with pytest.raises(ValueError, match="Mismatching shapes"):
        model = _MockModel(device="meta", func=shared_data_shape)
        model.initialize_shared_data()
        model.tie_shared_data()
