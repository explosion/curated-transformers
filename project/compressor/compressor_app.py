import typer
import srsly

import numpy as np

from spacy.util import ensure_path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import make_autoencoder, make_twinembeddings
from data import make_loader
from train import make_loss, training_loop


app = typer.Typer()


def load_model(config):
    model_config = config["model"]
    original_size = model_config["original_size"]
    ratio = model_config["ratio"]
    compress_size = int(original_size / ratio)
    if model_config["type"] == "autoencoder":
        num_hidden = model_config["num_hidden"]
        hidden_dims = np.linspace(original_size, compress_size, num_hidden + 2)
        hidden_dims = [int(x) for x in hidden_dims[1:-1]]
        model = make_autoencoder(
            activation=model_config["activation"],
            in_dim=original_size,
            out_dim=compress_size,
            normalize=model_config["normalize"],
            hidden_dims=hidden_dims,
            rezero=model_config["rezero"]
        )
    elif model_config["type"] == "twinembedding":
        model = make_twinembeddings(
            num_embeddings=model_config["num_embeddings"],
            embedding_dim=compress_size,
            out_dim=original_size
        )
    else:
        raise ValueError(
            "Possible values for model type are autoencoder and "
            f"twinembedding, but found {model_config['type']}"
        )
    return model


@app.command()
def init_config(
    data_path: str,
    config_path: str,
    model_type: str
):
    model_config = {
        "type": model_type,
        "original_size": None,
        "ratio": None
    }
    if model_type == "autoencoder":
        model_config["num_hidden"] = None
        model_config["activation"] = None
        model_config["rezero"] = None
        model_config["normalize"] = True
    elif model_type == "twinembedding":
        model_config["num_embeddings"] = None
    else:
        raise NotImplementedError
    loader_config = {
        "path": data_path,
        "batch_size": 512,
        "normalizer": None
    }
    optimizer_config = {
        "learning_rate": 0.001,
        "weight_decay": 0.01
    }
    scheduler_config = {
        "factor": 0.1,
        "patience": 2
    }
    loss_config = {
        "type": "mse",
        "reduction": "mean"
    }
    training_config = {
        "epochs": 100,
        "patience": 5
    }
    config = {
        "model": model_config,
        "loader": loader_config,
        "optimizer": optimizer_config,
        "scheduler": scheduler_config,
        "loss": loss_config,
        "training": training_config
    }
    srsly.write_yaml(config_path, config)


@app.command()
def compress(
    data_path: str,
    config_path: str,
    out_path: str
):
    config = srsly.read_yaml(config_path)
    model = load_model(config)
    model_type = config["model"]["type"]
    path = config["loader"]["path"]
    batch_size = config["loader"]["batch_size"]
    normalizer = config["loader"]["normalizer"]
    loader = make_loader(model_type, path, batch_size, normalizer)
    learning_rate = config["optimizer"]["learning_rate"]
    weight_decay = config["optimizer"]["weight_decay"]
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    factor = config["scheduler"]["factor"]
    patience = config["scheduler"]["patience"]
    scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    loss_type = config["loss"]["type"]
    reduction = config["loss"]["reduction"]
    if loss_type == "huber":
        delta = config["loss"]["delta"]
    else:
        delta = None
    loss_fn = make_loss(loss_type, reduction=reduction, delta=delta)
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    out_path = ensure_path(out_path)
    training_loop(
        model,
        loss_fn,
        loader,
        epochs,
        patience,
        optimizer,
        scheduler,
        out_path
    )


if __name__ == "__main__":
    app()
