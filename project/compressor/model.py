import os

from typing import Sequence, Union, Callable

import numpy as np

from tqdm import tqdm

from thinc.types import Floats2d
from torch import nn
from torch.utils.data import DataLoader
from spacy.util import ensure_path

from data import TransformerLoader, Vectors, collate_autoencoder
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


def _encode_vectors(
        arr: Floats2d,
        model,
        batch_size: int,
        collate_fn: Callable
):
    data = DataLoader(
        Vectors(arr),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    ),
    num_embeddings = len(data.dataset)
    emb = np.empty((num_embeddings, model.compressed_size))
    bottom = 0
    for batch in tqdm(data):
        X, Y = batch
        top = bottom + X.size()[0]
        emb[bottom:top] = model.encode(X)
        bottom = top
    return emb


def _encode_transformer(loader: TransformerLoader, model):
    ...


def serialize(
    model: Union[AutoEncoder, TwinEmbeddings],
    data: Union[DataLoader, TransformerLoader],
    path: str,
) -> None:
    """
    Save compressed embeddings and linear decoder
    to disk as numpy arrays.
    """
    os.mkdir(path)
    W = model.decoder.weight.detach().numpy()
    b = model.decoder.bias.detach().numpy()
    np.save(ensure_path(path) / "weights", W)
    np.save(ensure_path(path) / "bias", b)
    if isinstance(model, AutoEncoder):
        if isinstance(data, DataLoader):
            emb = _encode_vectors(
                data.dataset.vectors,
                model,
                data.batch_size,
                data.collate_fn
            )
            np.save(ensure_path(path) / "embs", emb)
        elif isinstance(data, TransformerLoader):
            word_pieces = _encode_vectors(
                data.word_pieces,
                model,
                data.batch_size,
                collate_autoencoder
            )
            positional = _encode_vectors(
                data.positional,
                model,
                data.batch_size,
                collate_autoencoder
            )
            token_type = _encode_vectors(
                data.token_type,
                model,
                data.batch_size,
                collate_autoencoder
            )
            np.save(ensure_path(path) / "word_pieces", word_pieces)
            np.save(ensure_path(path) / "positional", positional)
            np.save(ensure_path(path) / "token_type", token_type)
    elif isinstance(model, TwinEmbeddings):
        emb = model.embeddings.weight.detach().numpy()
        np.save(ensure_path(path) / "embs", emb)
    else:
        raise ValueError(f"Cannot serialize object {model}")


def deserialize(
    path: str,
    target_transformer: str
):
    """
    Load compressed embeddings and linear decoder
    from disk and stitch them into a transformer.
    """
    ...
