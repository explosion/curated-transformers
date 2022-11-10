import numpy as np
import torch

from typing import Callable
from thinc.types import Floats2d
from torch.utils.data import Dataset, DataLoader


def zscore(X: Floats2d) -> Floats2d:
    m = X.mean(axis=1)
    std = X.std(axis=1)
    return (X - m) / std


def l2norm(X: Floats2d) -> Floats2d:
    norm = np.linalg.norm(X, axis=1)
    return X / norm


def whiten(X: Floats2d) -> Floats2d:
    """
    Maybe I'll implement ZCA whitening, but
    I feel like its too expensive.
    """
    ...


NORMALIZERS = {
    "zscore": zscore,
    "l2norm": l2norm,
}


class Vectors(Dataset):
    def __init__(
        self,
        vectors: Floats2d,
        *,
        normalizer: Callable[[Floats2d], Floats2d] = None,
    ):
        if normalizer:
            vectors = normalizer(vectors)
        self.vectors = vectors

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, idx):
        mat = torch.tensor(self.vectors[idx], dtype=torch.float32)
        idx = torch.tensor(idx)
        return mat, idx


def collate_autoencoder(batch) -> torch.Tensor:
    matrices, _ = zip(*batch)
    X = torch.stack(matrices, 0)
    return X, X


def collate_twinembedding(batch):
    matrices, idx = zip(*batch)
    X = torch.stack(matrices, 0)
    return torch.as_tensor(idx), X


def make_loader(
    model_type: str,
    path: str,
    batch_size: int,
    normalizer: Callable[[Floats2d], Floats2d] = None
) -> DataLoader:
    X = np.load(path)
    assert X.ndim == 2
    if normalizer is not None:
        if normalizer not in NORMALIZERS:
            raise ValueError(f"Could not find normalizer {normalizer}")
        else:
            normalizer = NORMALIZERS[normalizer]
    data = Vectors(X, normalizer=normalizer)
    if model_type == "autoencoder":
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_autoencoder
        )
    elif model_type == "twinembedding":
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_twinembedding
        )
    else:
        raise NotImplementedError(
            f"Could not find model-type {model_type}"
        )
    return loader
