import pytest
import torch

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.models.pytorch.embeddings import RotaryEmbeddings
from curated_transformers.tests.util import torch_assertclose


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_rotary_embeddings_against_hf():
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    re = RotaryEmbeddings(768)
    hf_re = LlamaRotaryEmbedding(768)

    X = torch.rand(16, 12, 64, 768)
    re_cos, re_sin = re(X)
    hf_re_cos, hf_re_sin = hf_re(X, seq_len=X.shape[-2])

    torch_assertclose(re_cos, hf_re_cos)
    torch_assertclose(re_sin, hf_re_sin)


def test_rotary_embeddings_resize():
    re = RotaryEmbeddings(4, seq_len=8)
    assert re.cos.shape == (1, 1, 8, 4)
    assert re.sin.shape == (1, 1, 8, 4)
    X = torch.rand(4, 8, 16, 4)
    re(X)
    assert re.cos.shape == (1, 1, 16, 4)
    assert re.sin.shape == (1, 1, 16, 4)


def test_rotary_embeddings_small():
    re = RotaryEmbeddings(4)
    X = torch.rand(1, 2, 3, 4)
    re_cos, re_sin = re(X)
    torch_assertclose(
        re_cos,
        torch.tensor(
            [
                [
                    [
                        [1.000000, 1.000000, 1.000000, 1.000000],
                        [0.540302, 0.999950, 0.540302, 0.999950],
                        [-0.416147, 0.999800, -0.416147, 0.999800],
                    ]
                ]
            ]
        ),
    )
    torch_assertclose(
        re_sin,
        torch.tensor(
            [
                [
                    [
                        [0.000000, 0.000000, 0.000000, 0.000000],
                        [0.841471, 0.010000, 0.841471, 0.010000],
                        [0.909297, 0.019999, 0.909297, 0.019999],
                    ]
                ]
            ]
        ),
    )
