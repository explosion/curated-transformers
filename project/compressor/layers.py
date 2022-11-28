import torch

from torch import Tensor
from thinc.types import Floats2d, Floats1d
from torch import nn
from typing import Sequence, Optional
from collections import OrderedDict

from data import TransformerBatch
from dataclasses import dataclass


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
        self.layer_norm = nn.LayerNorm(self.encoder.out_dim)

    def encode(self, X):
        with torch.no_grad():
            self.eval()
            return self.encoder(X)

    def forward(self, X):
        # Make a LayerNorm on demand
        if isinstance(X, TransformerBatch):
            wp = self.encoder(X.word_pieces)
            pos = self.encoder(X.positional)
            typ = self.encoder(X.token_type)
            return self.decoder(self.layer_norm(wp + pos + typ))
        else:
            return self.decoder(self.layer_norm(self.encoder(X)))

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
