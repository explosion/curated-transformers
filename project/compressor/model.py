from typing import List, Union

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from spacy.util import ensure_path

from layers import AutoEncoder, ACTIVATIONS, MLP, TwinEmbeddings


def make_autoencoder(
        activation: Union[str, nn.Module],
        in_dim: int,
        out_dim: int,
        normalize: bool = True,
        hidden_dims: List[int] = [],
        *,
        rezero: bool = False
) -> AutoEncoder:
    if activation not in ACTIVATIONS:
        raise ValueError(f"Could not find activation {activation}")
    encoder = MLP(activation, in_dim, out_dim, normalize, hidden_dims, rezero)
    autoencoder = AutoEncoder(encoder)
    return autoencoder


def make_twinembeddings(num_embeddings, embedding_dim, out_dim):
    return TwinEmbeddings(num_embeddings, embedding_dim, out_dim)


def serialize(
    model: Union[AutoEncoder, TwinEmbeddings],
    data: DataLoader,
    path: str
) -> None:
    W = model.decoder.weight.detach().numpy()
    b = model.decoder.bias.detach().numpy()
    if isinstance(model, AutoEncoder):
        ...
    elif isinstance(model, TwinEmbeddings):
        emb = model.embeddings.weight.detach().numpy()
    else:
        raise ValueError(f"Cannot serialize object {model}")
    np.save(W, ensure_path(path) / "weights")
    np.save(b, ensure_path(path) / "bias")
    np.save(emb, ensure_path(path) / "embs")
