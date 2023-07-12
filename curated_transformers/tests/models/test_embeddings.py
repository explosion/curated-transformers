import pytest
import torch

from curated_transformers.layers.embeddings import RotaryEmbeddings

from ..compat import has_hf_transformers
from ..conftest import TORCH_DEVICES
from ..util import torch_assertclose


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_rotary_embeddings_against_hf(device):
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        rotate_half,
    )

    re = RotaryEmbeddings(768).to(device)
    hf_re = LlamaRotaryEmbedding(768, device=device)

    X = torch.rand(16, 12, 64, 768, device=device)
    Y = re(X)
    hf_re_cos, hf_re_sin = hf_re(X, seq_len=X.shape[-2])
    Y_hf = hf_re_cos * X + hf_re_sin * rotate_half(X)

    torch_assertclose(Y, Y_hf)


def test_rotary_embeddings_rejects_uneven_width():
    with pytest.raises(ValueError, match=r"must be even"):
        RotaryEmbeddings(5, seq_len=8)


@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_rotary_embeddings_resize(device):
    re = RotaryEmbeddings(4, seq_len=8).to(device)
    assert re.cos.shape == (8, 4)
    assert re.sin.shape == (8, 4)
    X = torch.rand(4, 8, 16, 4, device=device)
    re(X)
    assert re.cos.shape == (16, 4)
    assert re.sin.shape == (16, 4)

    positions = torch.tensor([[2, 1, 32], [1, 2, 0]], device=device).view([2, 1, 3])
    X = torch.rand(2, 2, 3, 4, device=device)
    re(X, positions=positions)
    assert re.cos.shape == (33, 4)
    assert re.sin.shape == (33, 4)


@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_rotary_embeddings_small(device):
    re = RotaryEmbeddings(4).to(device)
    X = torch.ones(1, 2, 3, 4, device=device)
    Y = re(X)
    torch_assertclose(
        Y,
        torch.tensor(
            [
                [
                    [
                        [1.000000, 1.000000, 1.000000, 1.000000],
                        [-0.301169, 0.989950, 1.381773, 1.009950],
                        [-1.325444, 0.979801, 0.493151, 1.019799],
                    ],
                    [
                        [1.000000, 1.000000, 1.000000, 1.000000],
                        [-0.301169, 0.989950, 1.381773, 1.009950],
                        [-1.325444, 0.979801, 0.493151, 1.019799],
                    ],
                ]
            ],
            device=device,
        ),
    )


@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_rotary_embeddings_positions_small(device):
    re = RotaryEmbeddings(4).to(device)
    X = torch.ones(2, 5, 3, 4, device=device)
    positions = torch.tensor([[2, 1, 0], [1, 2, 0]], device=device).view([2, 3])
    Y = re(X, positions=positions)
    torch_assertclose(
        Y,
        torch.tensor(
            [
                [
                    [
                        [-1.325444, 0.979801, 0.493151, 1.019799],
                        [-0.301169, 0.989950, 1.381773, 1.009950],
                        [1.000000, 1.000000, 1.000000, 1.000000],
                    ],
                ],
                [
                    [
                        [-0.301169, 0.989950, 1.381773, 1.009950],
                        [-1.325444, 0.979801, 0.493151, 1.019799],
                        [1.000000, 1.000000, 1.000000, 1.000000],
                    ],
                ],
            ],
            device=device,
        ).expand([2, 5, 3, 4]),
    )
