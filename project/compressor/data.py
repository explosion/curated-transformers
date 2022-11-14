import numpy
import torch
import spacy

from typing import Tuple, Callable

from thinc.types import Floats2d, ArrayXd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel


def get_vectors(path: str) -> Floats2d:
    """Load embeddings from path."""
    arr = numpy.load(path)
    assert arr.ndim == 2
    return arr


def get_spacy_vectors(model) -> Floats2d:
    """Load vectors from spaCy model."""
    nlp = spacy.load(model)
    return nlp.vocab.vectors.data


def get_hf_embeddings(name: str) -> Tuple[Floats2d, Floats2d, Floats2d]:
    """
    Load word-piece, positional and token-type
    embeddings with HuggingFace.
    """
    hf_model = AutoModel.from_pretrained(name)
    embedding_tensor = hf_model.embeddings.word_embeddings.weight
    position_tensor = hf_model.embeddings.position_embeddings.weight
    token_type_tensor = hf_model.embeddings.token_type_embeddings.weight
    embedding_matrix = embedding_tensor.detach().numpy()
    position_matrix = position_tensor.detach().numpy()
    token_type_matrix = token_type_tensor.detach().numpy()
    return embedding_matrix, position_matrix, token_type_matrix


class Vectors(Dataset):
    def __init__(
        self,
        vectors: Floats2d,
    ):
        self.vectors = vectors

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        mat = torch.tensor(self.vectors[idx], dtype=torch.float32)
        idx = torch.tensor(idx)
        return mat, idx


def cyclic_slice(arr: ArrayXd, bottom: int, top: int) -> ArrayXd:
    assert top > bottom
    size = arr.shape[0]
    idx = numpy.arange(bottom, top) % size
    return arr[idx]


# TODO Currently only works with the AutoEncoder
class TransformerLoader:
    def __init__(
        self,
        word_pieces: Floats2d,
        positional: Floats2d,
        token_type: Floats2d,
        *,
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        It shuffles the rows of all three matrices
        and yields rowwise sums. It considers a pass
        through the word-pieces as a full epoch.
        During the epoch each batch is different set
        of rows from the word-pieces summed together
        with postional and token_type embeddings.
        All arrays get reshuffled when the word-pieces
        are exhausted. The positional and token_type
        embeddings until reshuffling are indexed cyclically.
        """
        assert word_pieces.ndim == 2
        assert positional.ndim == 2
        assert token_type.ndim == 2
        assert word_pieces.shape[1] == positional.shape[1]
        assert positional.shape[1] == token_type.shape[1]

        self.word_pieces = word_pieces
        self.positional = positional
        self.token_type = token_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._n = word_pieces.shape[0]
        self.dim = token_type.shape[1]

    def __iter__(self):
        self._cursor = 0
        if self.shuffle:
            numpy.random.shuffle(self.word_pieces)
            numpy.random.shuffle(self.positional)
            numpy.random.shuffle(self.token_type)
        return self

    def __next__(self) -> Tuple[Floats2d, Floats2d]:
        # New epoch based on word-pieces.
        if self._cursor + self.batch_size >= self._n - 1:
            raise StopIteration
        else:
            wp_top = self._cursor + self.batch_size
            word_piece = self.word_pieces[self._cursor:wp_top]
            positional = cyclic_slice(
                self.positional, self._cursor, wp_top
            )
            token_type = cyclic_slice(
                self.token_type, self._cursor, wp_top
            )
            self._cursor = wp_top
            out = torch.tensor(
                word_piece + positional + token_type, dtype=torch.float32
            )
            return out, out


def collate_autoencoder(batch) -> torch.Tensor:
    matrices, _ = zip(*batch)
    X = torch.stack(matrices, 0)
    return X, X


def collate_twinembedding(batch):
    matrices, idx = zip(*batch)
    X = torch.stack(matrices, 0)
    return torch.as_tensor(idx), X


def make_vector_loader(
    path: str,
    model_type: str,
    batch_size: int,
) -> DataLoader:
    X = get_vectors(path)
    assert X.ndim == 2
    data = Vectors(X)
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


def make_transformer_loader(
    transformer: str,
    batch_size: int
) -> TransformerLoader:
    word_pieces, positional, token_type = get_hf_embeddings(transformer)
    loader = TransformerLoader(
        word_pieces,
        positional,
        token_type,
        batch_size=batch_size,
        shuffle=True
    )
    return loader
