import numpy as np
import torch
from torch.nn import LayerNorm
from .data import TransformerLoader 


#TODO this fails
def test_transformerloader():
    n, d, bs = 100, 20, 10
    wp = np.random.random(size=(n, d))
    pos = np.random.random(size=(n, d))
    typ = np.random.random(size=(n, d))
    layernorm = LayerNorm(d)
    loader = TransformerLoader(
        wp, pos, typ, layernorm, batch_size=bs
    )
    bottom = 0
    for batch in loader:
        X, Y = batch
        wp = X.word_pieces
        pos = X.positional
        typ = X.token_type
        normed = layernorm(wp + pos + typ)
        assert wp.shape == Y.shape
        assert pos.shape == Y.shape
        assert typ.shape == Y.shape
        assert torch.allclose(normed, Y)
