import pytest
import torch
from curated_transformers.generation.logits import (
    MaskTransform,
    TemperatureTransform,
    TopKTransform,
)

from ..util import torch_assertclose


def test_temperature_transform():
    logits = torch.rand((2, 10, 20), dtype=torch.float)
    transform = TemperatureTransform(temperature=1.0)
    torch_assertclose(transform(logits), logits)
    transform = TemperatureTransform(temperature=0.5)
    torch_assertclose(transform(logits), logits / 0.5)
    transform = TemperatureTransform(temperature=2.0)
    torch_assertclose(transform(logits), logits / 2.0)


def test_invalid_temperature_raises():
    with pytest.raises(ValueError, match=r"The temperature must be a non-zero"):
        TemperatureTransform(temperature=0.0)


def test_top_k_transform():
    logits = torch.tensor(
        [[4.0, 4.0, 3.5, 0.0, 3.5, 0.0, 3.5, 4.0]], dtype=torch.float32
    )
    for k in [1, 2, 3]:
        transform = TopKTransform(k=k)
        torch_assertclose(
            transform(logits),
            torch.tensor(
                [
                    [
                        4.0,
                        4.0,
                        -3.4028e38,
                        -3.4028e38,
                        -3.4028e38,
                        -3.4028e38,
                        -3.4028e38,
                        4.0,
                    ]
                ],
                dtype=torch.float32,
            ),
        )

    for k in [4, 5, 6]:
        transform = TopKTransform(k=k)
        torch_assertclose(
            transform(logits),
            torch.tensor(
                [
                    [
                        4.0,
                        4.0,
                        3.5,
                        -3.4028e38,
                        3.5,
                        -3.4028e38,
                        3.5,
                        4.0,
                    ]
                ],
                dtype=torch.float32,
            ),
        )
        transform = TopKTransform(k=k)

    for k in [0, 8]:
        transform = TopKTransform(k=k)
        torch_assertclose(
            transform(logits),
            torch.tensor(
                [
                    [
                        4.0,
                        4.0,
                        3.5,
                        0.0,
                        3.5,
                        0.0,
                        3.5,
                        4.0,
                    ]
                ],
                dtype=torch.float32,
            ),
        )


def test_mask_transform():
    logits = torch.tensor(
        [[[4.0, 4.0, 3.5, 0.0, 3.5, 0.0, 3.5, 4.0]]], dtype=torch.float32
    )

    transform = MaskTransform(classes_to_mask=[2, 4, 5])
    torch_assertclose(
        transform(logits),
        torch.tensor(
            [
                [
                    [
                        4.0,
                        4.0,
                        -3.4028e38,
                        0.0,
                        -3.4028e38,
                        -3.4028e38,
                        3.5,
                        4.0,
                    ]
                ]
            ],
            dtype=torch.float32,
        ),
    )

    transform = MaskTransform(classes_to_mask=[])
    torch_assertclose(transform(logits, inplace=False), logits)


def test_invalid_mask_transform_raises():
    with pytest.raises(ValueError, match="must be 1D"):
        transform = MaskTransform(classes_to_mask=[[1, 2]])

    with pytest.raises(ValueError, match="must be >= 0"):
        transform = MaskTransform(classes_to_mask=[-1, 0])

    with pytest.raises(ValueError, match="must be < .*, but got"):
        transform = MaskTransform(classes_to_mask=[8])
        transform(
            torch.tensor(
                [[4.0, 4.0, 3.5, 0.0, 3.5, 0.0, 3.5, 4.0]], dtype=torch.float32
            )
        )
