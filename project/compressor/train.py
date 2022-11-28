import shutil

import torch

from typing import Union

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from layers import AutoEncoder, TwinEmbeddings
from model import serialize


LOSSES = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "huber": nn.HuberLoss
}


def make_loss(name: str, *, reduction: int = "mean", delta: float = 1.0):
    if name not in LOSSES:
        raise ValueError(f"Could not find loss {name}")
    loss = LOSSES[name]
    if name == "huber":
        loss = loss(reduction=reduction, delta=delta)
    else:
        loss = loss(reduction=reduction)
    return loss


def training_loop(
    model: Union[AutoEncoder, TwinEmbeddings],
    loss_fn,
    data: DataLoader,
    epochs: int,
    patience: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    save_path: str
):
    """
    Training loop that is pursuing the minimization of
    the training error. After each epoch it runs through
    the entire training set and computes the average loss.
    After each epoch if the model has lower mean loss on
    the entire training set the model gets saved and the
    previous checkpoint is deleted.
    """
    model = model.to("cuda")
    def eval():
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(data):
                X, Y = batch
                X.to_cuda()
                Y = Y.to("cuda")
                Y_hat = model(X)
                total_loss += loss_fn(Y_hat, Y)
        model.train()
        return total_loss / (i + 1)

    no_improve = 0
    best_loss = float("inf")
    best_model = None
    print(f"Starting loss: {eval()}")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        pbar = tqdm(data, total=len(data))
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            X, Y = batch
            X.to_cuda()
            Y = Y.to("cuda")
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y)
            pbar.set_description(f"Loss: {loss}")
            loss.backward()
            optimizer.step()
        mean_loss = eval()
        print(f"Best loss on full data achieved: {mean_loss}")
    model = model.to("cpu")
    return model
