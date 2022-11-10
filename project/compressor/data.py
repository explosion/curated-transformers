import torch

from thinc.types import Floats2d
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel


def get_embedding(name):
    """
    Take the (sub)word embeddings of huggingface
    transformers as long as they are in
    model.embeddings.word_embeddings
    """
    hf_model = AutoModel.from_pretrained(name)
    embedding_tensor = hf_model.embeddings.word_embeddings.weight
    embedding_matrix = embedding_tensor.detach().numpy()
    return embedding_matrix


class Vectors(Dataset):
    def __init__(
        self,
        vectors: Floats2d,
    ):
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
    transformer: str,
    model_type: str,
    batch_size: int,
) -> DataLoader:
    X = get_embedding(transformer)
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
