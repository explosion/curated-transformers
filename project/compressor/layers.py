from torch import nn
from typing import List


ACTIVATIONS = {
    "linear": nn.Identity(),
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "swish": nn.SiLU()
}


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
    stack = []
    linear = nn.Linear(nI, nO)
    stack.append(Residual(linear, rezero))
    stack.append(act)
    if norm:
        layernorm = nn.LayerNorm(nI)
        stack.append(layernorm)
    return nn.Sequential(stack)


class Residual(nn.Module):
    def __init__(self, layer: nn.Module, *, rezero: bool = False):
        self.layer = layer
        self.rezero = rezero
        if rezero:
            self.alpha = nn.Parameter(0.0)

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
        hidden_dims: List[int] = [],
        *,
        rezero: bool = False
    ):
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
            stack = []
            input_layer = make_layer(
                in_dim, hidden_dims[0], activation, normalize, rezero=rezero
            )
            output_layer = make_layer(
                hidden_dims[-1], out_dim, activation, normalize, rezero=rezero
            )
            stack.append(input_layer)
            if self.depth > 2:
                nI = self.in_dim
                for i in range(0, len(hidden_dims)):
                    nO = hidden_dims[i]
                    layer = make_layer(
                        nI, nO, activation, normalize, rezero=rezero
                    )
                    nI = nO
                    stack.append(layer)
            stack.append(output_layer)
            self.network = nn.Sequential(stack)

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
        self.encoder = encoder
        self.decoder = nn.Linear(encoder.out_dim, encoder.in_dim)

    def encode(self, X):
        return self.encoder(X).detach().numpy()

    def forward(self, X):
        return self.decoder(self.encoder(X))


class TwinEmbeddings:
    """
    An embedding table coupled with a linear layer.
    It allows to train embeddings that mimic another
    embedding table.
    """
    def __init__(self, num_embeddings, embedding_dim, out_dim):
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, out_dim)

    def encoder(self, idx):
        return self.embeddings(idx).detach().numpy()

    def forward(self, idx):
        return self.decoder(self.embeddings(idx))
