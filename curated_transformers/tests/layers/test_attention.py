import pytest
import torch

from curated_transformers.layers.attention import (
    _TORCH_SDP,
    AttentionLinearBiases,
    AttentionMask,
    enable_torch_sdp,
)
from curated_transformers.models.bert.encoder import BERTEncoder
from curated_transformers.models.gpt_neox.decoder import GPTNeoXDecoder

from ..compat import has_hf_transformers
from ..conftest import TORCH_DEVICES
from ..utils import torch_assertclose

N_PIECES = 1024


def test_context_manager():
    # Check default.
    assert not _TORCH_SDP.get()

    # Check context manager default.
    with enable_torch_sdp():
        assert _TORCH_SDP.get()

    # Check that the original value is restored.
    assert not _TORCH_SDP.get()

    # Check with explicit argument.
    with enable_torch_sdp(True):
        assert _TORCH_SDP.get()
    with enable_torch_sdp(False):
        assert not _TORCH_SDP.get()


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_torch_sdp(torch_device):
    model = BERTEncoder.from_hf_hub(
        name="explosion-testing/bert-test", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, N_PIECES, (2, 10), device=torch_device)
    mask = torch.ones_like(X, dtype=torch.bool)
    with torch.no_grad():
        Y = model(X, AttentionMask(mask)).last_hidden_layer_state
        with enable_torch_sdp():
            Y_sdp = model(X, AttentionMask(mask)).last_hidden_layer_state
    torch_assertclose(Y, Y_sdp)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_torch_sdp_mask(torch_device):
    model = BERTEncoder.from_hf_hub(
        name="explosion-testing/bert-test", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, N_PIECES, (2, 10), device=torch_device)
    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = model(
            X, attention_mask=AttentionMask(mask)
        ).last_hidden_layer_state * mask.unsqueeze(-1)
        with enable_torch_sdp():
            Y_sdp = model(
                X, attention_mask=AttentionMask(mask)
            ).last_hidden_layer_state * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_sdp)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_torch_sdp_causal(torch_device):
    model = GPTNeoXDecoder.from_hf_hub(
        name="trl-internal-testing/tiny-random-GPTNeoXForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, N_PIECES, (2, 10), device=torch_device)
    mask = torch.ones_like(X, dtype=torch.bool)
    with torch.no_grad():
        Y = model(X, AttentionMask(mask)).last_hidden_layer_state
        with enable_torch_sdp():
            Y_sdp = model(X, AttentionMask(mask)).last_hidden_layer_state
    torch_assertclose(Y, Y_sdp)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_torch_sdp_causal_with_mask(torch_device):
    model = GPTNeoXDecoder.from_hf_hub(
        name="trl-internal-testing/tiny-random-GPTNeoXForCausalLM", device=torch_device
    )
    model.eval()

    torch.manual_seed(0)
    X = torch.randint(0, N_PIECES, (2, 10), device=torch_device)
    mask = torch.rand((2, 10), dtype=torch.float, device=torch_device) < 0.5
    with torch.no_grad():
        Y = model(
            X, attention_mask=AttentionMask(mask)
        ).last_hidden_layer_state * mask.unsqueeze(-1)
        with enable_torch_sdp():
            Y_sdp = model(
                X, attention_mask=AttentionMask(mask)
            ).last_hidden_layer_state * mask.unsqueeze(-1)
    torch_assertclose(Y, Y_sdp)


@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_attention_linear_biases(torch_device):
    pow2_slopes = AttentionLinearBiases(
        n_attention_heads=8, is_causal=False, is_inverted=False
    ).slopes
    torch_assertclose(
        pow2_slopes.to(device=torch_device),
        torch.tensor(
            [
                [
                    [[0.5]],
                    [[0.25]],
                    [[0.125]],
                    [[0.0625]],
                    [[0.03125]],
                    [[0.015625]],
                    [[0.0078125]],
                    [[0.00390625]],
                ]
            ],
            device=torch_device,
        ),
    )
    non_pow2_slopes = AttentionLinearBiases(
        n_attention_heads=12, is_causal=False, is_inverted=False
    ).slopes
    torch_assertclose(
        non_pow2_slopes.to(device=torch_device),
        torch.tensor(
            [
                [
                    [[0.5]],
                    [[0.25]],
                    [[0.125]],
                    [[0.0625]],
                    [[0.03125]],
                    [[0.015625]],
                    [[0.0078125]],
                    [[0.00390625]],
                    [[0.7071067811865476]],
                    [[0.35355339059327384]],
                    [[0.17677669529663692]],
                    [[0.08838834764831849]],
                ]
            ],
            device=torch_device,
        ),
    )

    alibi_causal = AttentionLinearBiases(
        n_attention_heads=4, is_causal=True, is_inverted=False
    )
    torch_assertclose(
        alibi_causal(attention_scores=torch.zeros((1, 4, 1, 3), device=torch_device)),
        torch.tensor(
            [
                [
                    [[-0.5000, -0.2500, 0.0000]],
                    [[-0.1250, -0.0625, 0.0000]],
                    [[-0.03125, -0.015625, 0.0000]],
                    [[-0.0078125, -0.00390625, 0.0000]],
                ]
            ],
            device=torch_device,
        ),
    )

    alibi_non_causal = AttentionLinearBiases(
        n_attention_heads=4, is_causal=False, is_inverted=False
    )
    torch_assertclose(
        alibi_non_causal(
            attention_scores=torch.zeros((1, 4, 3, 3), device=torch_device)
        ),
        torch.tensor(
            [
                [
                    [
                        [0.0000, -0.2500, -0.5000],
                        [-0.2500, 0.0000, -0.2500],
                        [-0.5000, -0.2500, 0.0000],
                    ],
                    [
                        [0.0000, -0.0625, -0.1250],
                        [-0.0625, 0.0000, -0.0625],
                        [-0.1250, -0.0625, 0.0000],
                    ],
                    [
                        [0.0000, -0.015625, -0.03125],
                        [-0.015625, 0.0000, -0.015625],
                        [-0.03125, -0.015625, 0.0000],
                    ],
                    [
                        [0.0000, -0.00390625, -0.0078125],
                        [-0.00390625, 0.0000, -0.00390625],
                        [-0.0078125, -0.00390625, 0.0000],
                    ],
                ]
            ],
            device=torch_device,
        ),
    )

    alibi_causal_inverted = AttentionLinearBiases(
        n_attention_heads=4, is_causal=True, is_inverted=True
    )
    torch_assertclose(
        alibi_causal_inverted(
            attention_scores=torch.zeros((1, 4, 1, 3), device=torch_device)
        ),
        torch.tensor(
            [
                [
                    [[0.0000, 0.2500, 0.5000]],
                    [[0.0000, 0.0625, 0.1250]],
                    [[0.0000, 0.015625, 0.03125]],
                    [[0.0000, 0.00390625, 0.0078125]],
                ]
            ],
            device=torch_device,
        ),
    )

    alibi_non_causal_inverted = AttentionLinearBiases(
        n_attention_heads=4, is_causal=False, is_inverted=True
    )
    torch_assertclose(
        alibi_non_causal_inverted(
            attention_scores=torch.zeros((1, 4, 3, 3), device=torch_device)
        ),
        torch.tensor(
            [
                [
                    [
                        [0.5000, 0.2500, 0.0000],
                        [0.2500, 0.5000, 0.2500],
                        [0.0000, 0.2500, 0.5000],
                    ],
                    [
                        [0.1250, 0.0625, 0.0000],
                        [0.0625, 0.1250, 0.0625],
                        [0.0000, 0.0625, 0.1250],
                    ],
                    [
                        [0.03125, 0.015625, 0.0000],
                        [0.015625, 0.03125, 0.015625],
                        [0.0000, 0.015625, 0.03125],
                    ],
                    [
                        [0.0078125, 0.00390625, 0.0000],
                        [0.00390625, 0.0078125, 0.00390625],
                        [0.0000, 0.00390625, 0.0078125],
                    ],
                ]
            ],
            device=torch_device,
        ),
    )
