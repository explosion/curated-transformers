import torch

from torch import Tensor
from thinc.types import Floats2d, Floats1d
from torch import nn
from typing import Sequence, Optional
from collections import OrderedDict

from data import TransformerBatch


ACTIVATIONS = {
    "linear": nn.Identity(),
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "swish": nn.SiLU()
}


def init_layer(layer: nn.Module):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)


def make_layer(
    nI: int,
    nO: int,
    act: nn.Module,
    norm: bool,
    *,
    rezero: bool = False
) -> nn.Module:
    """
    Make a Residual Linear + Activation layer with a potential LayerNorm.
    """
    stack = OrderedDict()
    linear = nn.Linear(nI, nO)
    residual = nI == nO
    if residual:
        stack["residual"] = Residual(linear, rezero=rezero)
    else:
        stack["dense"] = linear
    stack["activation"] = act
    if norm:
        stack["norm"] = nn.LayerNorm(nO)
    return nn.Sequential(stack)


class Residual(nn.Module):
    def __init__(self, layer: nn.Module, *, rezero: bool = False):
        super(Residual, self).__init__()
        self.layer = layer
        self.rezero = rezero
        if rezero:
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, X):
        if self.rezero:
            return X + self.alpha * self.layer(X)
        else:
            return X + self.layer(X)


class MLP(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        in_dim: int,
        out_dim: int,
        normalize: bool = True,
        hidden_dims: Sequence[int] = [],
        *,
        rezero: bool = False
    ):
        super(MLP, self).__init__()
        self.depth = len(hidden_dims) + 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        if isinstance(activation, str):
            activation = ACTIVATIONS[activation]
        if self.depth == 1:
            self.network = make_layer(
                in_dim, out_dim, activation, normalize, rezero=rezero
            )
        else:
            stack = OrderedDict()
            input_layer = make_layer(
                in_dim, hidden_dims[0], activation, normalize, rezero=rezero
            )
            output_layer = make_layer(
                hidden_dims[-1], out_dim, activation, normalize, rezero=rezero
            )
            stack["input"] = input_layer
            if self.depth > 2:
                nI = hidden_dims[0]
                for i in range(1, len(hidden_dims)):
                    nO = hidden_dims[i]
                    layer = make_layer(
                        nI, nO, activation, normalize, rezero=rezero
                    )
                    nI = nO
                    stack[f"hidden_{i+1}"] = layer
            stack["output"] = output_layer
            self.network = nn.Sequential(stack)
            self.network.apply(init_layer)

    def forward(self, X):
        assert X.ndim == 2
        assert X.shape[1] == self.in_dim
        return self.network(X)


class AutoEncoder(nn.Module):
    """
    An autoencoder with potentially deep MLP with
    residual/rezero connections and optional layernorm
    as encoder and a single layer linear decoder.
    """
    def __init__(self, encoder: MLP):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(encoder.out_dim, encoder.in_dim)
        input(self.decoder)
        input(self.decoder.weight.data.shape)

    def encode(self, X):
        return self.encoder(X).detach().numpy()

    def forward(self, X):
        if isinstance(X, TransformerBatch):
            wp = self.encoder(X.word_pieces)
            pos = self.encoder(X.positional)
            typ = self.encoder(X.token_type)
            return self.decoder(wp + pos + typ)
        else:
            return self.decoder(self.encoder(X))

    @property
    def compressed_size(self):
        return self.encoder.out_dim


# TODO doesn't work with the transformer atm.
class TwinEmbeddings(nn.Module):
    """
    An embedding table coupled with a linear layer.
    It allows to train embeddings that mimic another
    embedding table.
    """
    def __init__(self, num_embeddings, embedding_dim, out_dim):
        super(TwinEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, out_dim)

    def encoder(self, idx):
        if isinstance(idx, TransformerBatch):
            raise NotImplementedError(
                "Cannot run transformer with TwinEmbeddings"
            )
        return self.embeddings(idx).detach().numpy()

    def forward(self, idx):
        if isinstance(idx, TransformerBatch):
            raise NotImplementedError(
                "Cannot run transformer with TwinEmbeddings"
            )
        return self.decoder(self.embeddings(idx))

    @property
    def compressed_size(self):
        return self.embedding_dim


# FIXME Dirty ass patch copy/paste BertEmbeddings frankenlayer
class ProjectedBertEmbeddings(nn.Module):
    def __init__(
        self,
        word_embeddings: Floats2d,
        token_type_embeddings: Floats2d,
        position_embeddings: Floats2d,
        W: Floats2d,
        b: Floats1d
    ) -> None:
        super().__init__()

        self.word_embeddings = torch.nn.Embedding(
            num_embeddings=word_embeddings.shape[0],
            embedding_dim=word_embeddings.shape[1],
            _weight=torch.tensor(word_embeddings, dtype=torch.float32)
        )
        self.token_type_embeddings = torch.nn.Embedding(
            num_embeddings=token_type_embeddings.shape[0],
            embedding_dim=token_type_embeddings.shape[1],
            _weight=torch.tensor(token_type_embeddings, dtype=torch.float32)
        )
        self.position_embeddings = torch.nn.Embedding(
            num_embeddings=position_embeddings.shape[0],
            embedding_dim=position_embeddings.shape[1],
            _weight=torch.tensor(position_embeddings, dtype=torch.float32)
        )
        self.projection = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor(W, dtype=torch.float32))
            self.projection.bias.copy_(torch.tensor(b, dtype=torch.float32))

    def _get_position_ids(self, x: Tensor) -> Tensor:
        return torch.arange(x.shape[1], device=x.device).expand(1, -1)

    def _get_token_type_ids(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if token_type_ids is None:
            token_type_ids = self._get_token_type_ids(input_ids)
        if position_ids is None:
            position_ids = self._get_position_ids(input_ids)

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embedding_sum = input_embeddings
        embedding_sum += token_type_embeddings
        embedding_sum += position_embeddings
        return self.projection(embedding_sum)
