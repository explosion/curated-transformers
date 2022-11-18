import numpy
import torch
import spacy
from wasabi import msg
from thinc.api import LayerNorm as ThincNorm
from torch import nn

from dataclasses import dataclass
from math import floor
from typing import Tuple

from thinc.types import Floats2d, Floats1d, ArrayXd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from curated_transformers.models.roberta.embeddings import RobertaEmbeddings


def layernorm2thinc(layernorm: nn.LayerNorm) -> ThincNorm:
    """
    Converts PyTorch LayerNorm to thinc LayerNorm.
    """
    W = layernorm.weight.detach().numpy()
    b = layernorm.bias.detach().numpy()
    norm = ThincNorm(b.shape[0])
    norm.initialize()
    norm.set_param("G", W)
    norm.set_param("b", b)
    return norm


def array2embedding(array: Floats2d) -> nn.Embedding:
    embedding = nn.Embedding(
        num_embeddings=array.shape[0],
        embedding_dim=array.shape[1],
        _weight=torch.tensor(array, dtype=torch.float32)
    )
    return embedding


def arrays2linear(weight: Floats2d, bias: Floats1d) -> nn.Linear:
    W = torch.tensor(weight, dtype=torch.float32)
    b = torch.tensor(bias, dtype=torch.float32)
    linear = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        linear.weight.copy_(W)
        linear.bias.copy_(b)
    return linear


def arrays2layernorm(weight: Floats1d, bias: Floats1d)-> nn.LayerNorm:
    weight = torch.tensor(weight, dtype=torch.float32)
    bias = torch.tensor(bias, dtype=torch.float32)
    layernorm = nn.LayerNorm(weight.shape[0])
    with torch.no_grad():
        layernorm.weight.copy_(weight)
        layernorm.bias.copy_(bias)
    return layernorm


def get_vectors(path: str) -> Floats2d:
    """Load embeddings from path."""
    arr = numpy.load(path)
    assert arr.ndim == 2
    return arr


def get_spacy_vectors(model) -> Floats2d:
    """Load vectors from spaCy model."""
    nlp = spacy.load(model)
    return nlp.vocab.vectors.data


def get_hf_transformer(name: str) -> Tuple[Floats2d, Floats2d, Floats2d]:
    """Word-piece, positional and token-type embeddings from HuggingFace."""
    hf_model = AutoModel.from_pretrained(name)
    embedding_tensor = hf_model.embeddings.word_embeddings.weight
    position_tensor = hf_model.embeddings.position_embeddings.weight
    token_type_tensor = hf_model.embeddings.token_type_embeddings.weight
    embedding_matrix = embedding_tensor.detach().numpy()
    position_matrix = position_tensor.detach().numpy()
    token_type_matrix = token_type_tensor.detach().numpy()
    return embedding_matrix, position_matrix, token_type_matrix


def get_curated_transformer(model: str) -> Tuple[Floats2d, Floats2d, Floats2d]:
    """
    Word-piece, positional and token-type
    embeddings from curated-transformers model.
    """
    nlp = spacy.load(model)
    if not nlp.has_pipe("transformer"):
        raise ValueError(
            f"Could not find transformer in the pipeline {model}"
        )
    trf_pipe = nlp.get_pipe("transformer")
    trf_model = trf_pipe.model.get_ref("transformer")
    trf_pytorch = trf_model.shims[0]._model
    embeddings = trf_pytorch.embeddings
    if isinstance(embeddings, RobertaEmbeddings):
        embeddings = embeddings.inner
    embedding_tensor = embeddings.word_embeddings.weight
    position_tensor = embeddings.position_embeddings.weight
    token_type_tensor = embeddings.token_type_embeddings.weight
    embedding_matrix = embedding_tensor.detach().numpy()
    position_matrix = position_tensor.detach().numpy()
    token_type_matrix = token_type_tensor.detach().numpy()
    layernorm = layernorm2thinc(embeddings.layer_norm)
    return embedding_matrix, position_matrix, token_type_matrix, layernorm


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


@dataclass
class TransformerBatch:
    word_pieces: Floats2d
    positional: Floats2d
    token_type: Floats2d


# TODO Currently only works with the AutoEncoder
class TransformerLoader:
    def __init__(
        self,
        word_pieces: Floats2d,
        positional: Floats2d,
        token_type: Floats2d,
        layernorm: ThincNorm,
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
        msg.good(f"Word-pieces shape {word_pieces.shape}")
        msg.good(f"Positional shape {positional.shape}")
        msg.good(f"Token-type shape {token_type.shape}")
        self.word_pieces = word_pieces
        self.positional = positional
        self.token_type = token_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._n = word_pieces.shape[0]
        self.dim = token_type.shape[1]
        self.layernorm = layernorm

    def __len__(self):
        return floor(self._n / self.batch_size)

    def __iter__(self):
        self._cursor = 0
        if self.shuffle:
            numpy.random.shuffle(self.word_pieces)
            numpy.random.shuffle(self.positional)
            numpy.random.shuffle(self.token_type)
        return self

    def __next__(self) -> Tuple[Floats2d, Floats2d]:
        if self._cursor == self._n - 1:
            raise StopIteration
        wp_top = min(self._cursor + self.batch_size, self._n - 1)
        word_piece = self.word_pieces[self._cursor:wp_top]
        positional = cyclic_slice(
            self.positional, self._cursor, wp_top
        )
        token_type = cyclic_slice(
            self.token_type, self._cursor, wp_top
        )
        self._cursor = wp_top
        X = TransformerBatch(
            torch.tensor(word_piece, dtype=torch.float32),
            torch.tensor(positional, dtype=torch.float32),
            torch.tensor(token_type, dtype=torch.float32)
        )
        Y, _ = self.layernorm(word_piece + positional + token_type, False)
        return X, torch.tensor(Y, dtype=torch.float32)


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
    batch_size: int,
    *,
    source: str = "curated"
) -> TransformerLoader:
    assert source in {"curated", "hf"}
    if source == "curated":
        wp, pos, typ, norm = get_curated_transformer(transformer)
    else:
        wp, pos, typ = get_hf_transformer(transformer)
    loader = TransformerLoader(
        wp,
        pos,
        typ,
        norm,
        batch_size=batch_size,
        shuffle=True
    )
    return loader
