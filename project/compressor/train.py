import torch

from typing import Union

from tqdm import tqdm
from torch import nn

from layers import AutoEncoder, TwinEmbeddings
from torch.utils.data import DataLoader


LOSSES = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "huber": nn.HuberLoss
}


def make_loss(name: str, reduction: str = "mean", **kwargs):
    if name not in LOSSES:
        raise ValueError(f"Could not find loss {name}")
    loss = LOSSES[name]
    loss = loss(reduction=reduction, **kwargs)
    return loss


def train(
        model: Union[AutoEncoder, TwinEmbeddings],
        loss_fn,
        data: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler=None,
        full_eval_freq: int = 1
):
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(data)
        for step, batch in enumerate(pbar, total=len(data)):
            optimizer.zero_grad()
            X, Y = batch
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y)
            pbar.set_description(f"Step {step}, loss: {loss}")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if epoch % full_eval_freq == 0:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for i, batch in enumerate(data):
                    X, Y = batch
                    Y_hat = model(X)
                    total_loss += loss_fn(Y_hat, Y)
                print(f"Total loss on full data: {total_loss / i}")
