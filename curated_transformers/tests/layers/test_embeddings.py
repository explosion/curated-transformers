import pytest
import torch

from curated_transformers.layers.embeddings import (
    RotaryEmbeddings,
    SinusoidalPositionalEmbedding,
)

from ..compat import has_hf_transformers
from ..conftest import TORCH_DEVICES
from ..utils import torch_assertclose


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
    positions = torch.arange(X.shape[2], device=device).view([1, -1])
    hf_re_cos, hf_re_sin = hf_re(X, positions)
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


def test_sinusoidal_embeddings_without_norm():
    embeddings = SinusoidalPositionalEmbedding(width=6, max_len=512, normalize=False)
    positions = embeddings(torch.ones(4, 20))
    torch_assertclose(
        positions,
        torch.tensor(
            [
                [0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000],
                [0.841471, 0.540302, 0.046399, 0.998923, 0.002154, 0.999998],
                [0.909297, -0.416147, 0.092699, 0.995694, 0.004309, 0.999991],
                [0.141120, -0.989992, 0.138798, 0.990321, 0.006463, 0.999979],
                [-0.756802, -0.653644, 0.184599, 0.982814, 0.008618, 0.999963],
                [-0.958924, 0.283662, 0.230002, 0.973190, 0.010772, 0.999942],
                [-0.279415, 0.960170, 0.274909, 0.961470, 0.012926, 0.999916],
                [0.656987, 0.753902, 0.319225, 0.947679, 0.015080, 0.999886],
                [0.989358, -0.145500, 0.362852, 0.931847, 0.017235, 0.999851],
                [0.412118, -0.911130, 0.405699, 0.914007, 0.019389, 0.999812],
                [-0.544021, -0.839072, 0.447671, 0.894198, 0.021543, 0.999768],
                [-0.999990, 0.004426, 0.488679, 0.872464, 0.023697, 0.999719],
                [-0.536573, 0.843854, 0.528634, 0.848850, 0.025850, 0.999666],
                [0.420167, 0.907447, 0.567451, 0.823407, 0.028004, 0.999608],
                [0.990607, 0.136737, 0.605045, 0.796191, 0.030158, 0.999545],
                [0.650288, -0.759688, 0.641336, 0.767260, 0.032311, 0.999478],
                [-0.287903, -0.957659, 0.676246, 0.736676, 0.034464, 0.999406],
                [-0.961397, -0.275163, 0.709698, 0.704506, 0.036617, 0.999329],
                [-0.750987, 0.660317, 0.741623, 0.670817, 0.038770, 0.999248],
                [0.149877, 0.988705, 0.771949, 0.635684, 0.040923, 0.999162],
            ]
        ),
    )


def test_sinusoidal_embeddings_with_norm():
    embeddings = SinusoidalPositionalEmbedding(width=6, max_len=512)
    positions = embeddings(torch.ones(4, 20))
    torch_assertclose(
        positions,
        torch.tensor(
            [
                [0.000000, 0.577350, 0.000000, 0.577350, 0.000000, 0.577350],
                [0.485823, 0.311944, 0.026789, 0.576728, 0.001244, 0.577349],
                [0.524983, -0.240262, 0.053520, 0.574864, 0.002488, 0.577345],
                [0.081476, -0.571572, 0.080135, 0.571762, 0.003732, 0.577338],
                [-0.436940, -0.377381, 0.106578, 0.567428, 0.004975, 0.577329],
                [-0.553635, 0.163772, 0.132792, 0.561872, 0.006219, 0.577317],
                [-0.161321, 0.554355, 0.158719, 0.555105, 0.007463, 0.577302],
                [0.379311, 0.435266, 0.184304, 0.547143, 0.008707, 0.577285],
                [0.571206, -0.084004, 0.209493, 0.538002, 0.009950, 0.577265],
                [0.237937, -0.526041, 0.234230, 0.527702, 0.011194, 0.577242],
                [-0.314091, -0.484438, 0.258463, 0.516266, 0.012438, 0.577216],
                [-0.577345, 0.002555, 0.282139, 0.503717, 0.013681, 0.577188],
                [-0.309791, 0.487199, 0.305207, 0.490084, 0.014925, 0.577157],
                [0.242584, 0.523915, 0.327618, 0.475394, 0.016168, 0.577124],
                [0.571927, 0.078945, 0.349323, 0.459681, 0.017411, 0.577088],
                [0.375444, -0.438606, 0.370276, 0.442978, 0.018655, 0.577049],
                [-0.166221, -0.552905, 0.390431, 0.425320, 0.019898, 0.577007],
                [-0.555063, -0.158866, 0.409745, 0.406746, 0.021141, 0.576963],
                [-0.433583, 0.381234, 0.428176, 0.387297, 0.022384, 0.576916],
                [0.086532, 0.570829, 0.445685, 0.367012, 0.023627, 0.576867],
            ]
        ),
    )
