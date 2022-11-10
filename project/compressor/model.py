import os

from typing import Sequence, Union

import numpy as np

from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader
from spacy.util import ensure_path

from layers import AutoEncoder, ACTIVATIONS, MLP, TwinEmbeddings


def make_autoencoder(
        activation: Union[str, nn.Module],
        in_dim: int,
        out_dim: int,
        normalize: bool = True,
        hidden_dims: Sequence[int] = [],
        *,
        rezero: bool = False
) -> AutoEncoder:
    if activation not in ACTIVATIONS:
        raise ValueError(f"Could not find activation {activation}")
    encoder = MLP(
        activation, in_dim, out_dim, normalize, hidden_dims, rezero=rezero
    )
    autoencoder = AutoEncoder(encoder)
    return autoencoder


def make_twinembeddings(num_embeddings, embedding_dim, out_dim):
    return TwinEmbeddings(num_embeddings, embedding_dim, out_dim)


def serialize(
    model: Union[AutoEncoder, TwinEmbeddings],
    data: DataLoader,
    path: str,
) -> None:
    """
    Save compressed embeddings and linear decoder
    to disk as numpy arrays.
    """
    W = model.decoder.weight.detach().numpy()
    b = model.decoder.bias.detach().numpy()
    if isinstance(model, AutoEncoder):
        num_embeddings = len(data.dataset)
        emb = np.empty((num_embeddings, model.compressed_size))
        bottom = 0
        for batch in tqdm(data):
            X, Y = batch
            top = bottom + X.size()[0]
            emb[bottom:top] = model.encode(X)
            bottom = top
    elif isinstance(model, TwinEmbeddings):
        emb = model.embeddings.weight.detach().numpy()
    else:
        raise ValueError(f"Cannot serialize object {model}")
    os.mkdir(path)
    np.save(ensure_path(path) / "weights", W)
    np.save(ensure_path(path) / "bias", b)
    np.save(ensure_path(path) / "embs", emb)


def deserialize(
    path: str,
    target_transformer: str
):
    """
    Load compressed embeddings and linear decoder
    from disk and stitch them into a transformer.
    """
    ...
