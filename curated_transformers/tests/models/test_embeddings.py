import pytest
import torch

from curated_transformers._compat import has_hf_transformers
from curated_transformers.models.embeddings import RotaryEmbeddings
from curated_transformers.tests.util import torch_assertclose


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_rotary_embeddings_against_hf():
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        rotate_half,
    )

    re = RotaryEmbeddings(768)
    hf_re = LlamaRotaryEmbedding(768)

    X = torch.rand(16, 12, 64, 768)
    Y = re(X)
    hf_re_cos, hf_re_sin = hf_re(X, seq_len=X.shape[-2])
    Y_hf = hf_re_cos * X + hf_re_sin * rotate_half(X)

    torch_assertclose(Y, Y_hf)


def test_rotary_embeddings_rejects_uneven_width():
    with pytest.raises(ValueError, match=r"must be even"):
        RotaryEmbeddings(5, seq_len=8)


def test_rotary_embeddings_resize():
    re = RotaryEmbeddings(4, seq_len=8)
    assert re.cos.shape == (8, 4)
    assert re.sin.shape == (8, 4)
    X = torch.rand(4, 8, 16, 4)
    re(X)
    assert re.cos.shape == (16, 4)
    assert re.sin.shape == (16, 4)


def test_rotary_embeddings_small():
    re = RotaryEmbeddings(4)
    X = torch.ones(1, 2, 3, 4)
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
            ]
        ),
    )
