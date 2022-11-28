import typer
import srsly
import spacy
import torch
import numpy as np

from functools import partial

from spacy.util import ensure_path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from curated_transformers.models.roberta.embeddings import RobertaEmbeddings
from curated_transformers.models.bert.embeddings import BertEmbeddings
from curated_transformers.models.bert.config import BertEmbeddingConfig, BertLayerConfig
from model import make_autoencoder, make_twinembeddings
from data import make_transformer_loader, array2embedding, arrays2linear, arrays2layernorm
from data import get_curated_transformer, collate_autoencoder, Vectors
from train import make_loss, training_loop


app = typer.Typer()


def load_model(config):
    model_config = config["model"]
    original_size = model_config["original_size"]
    ratio = model_config["ratio"]
    compress_size = int(original_size / ratio)
    if model_config["type"] == "autoencoder":
        num_hidden = model_config["num_hidden"]
        residual = model_config["residual"]
        if residual:
            hidden_dims = [original_size for _ in range(num_hidden)]
        else:
            hidden_dims = np.linspace(
                original_size, compress_size, num_hidden + 2
            )
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
    transformer: str,
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
        "path": transformer,
        "batch_size": 512,
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
def compress_transformer(
    config_path: str,
    model_path: str,
    output_path: str
):
    config = srsly.read_yaml(config_path)
    nlp = spacy.load(model_path)
    trf_pipe = nlp.get_pipe("transformer")
    trf_model = trf_pipe.model.get_ref("transformer")
    trf_pytorch = trf_model.shims[0]._model
    embeddings = trf_pytorch.embeddings
    if isinstance(trf_pytorch.embeddings, RobertaEmbeddings):
        embeddings = embeddings.inner
    model_type = config["model"]["type"]
    if model_type != "autoencoder":
        raise NotImplementedError(
            "Transformer embedding compression, currently"
            "only works with the autoencoder"
        )
    transformer = config["loader"]["curated-transformers"]
    source = "curated"
    batch_size = config["loader"]["batch_size"]
    loader = make_transformer_loader(transformer, batch_size, source=source)
    config["model"]["original_size"] = loader.dim
    model = load_model(config)
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
    output_path = ensure_path(output_path)
    train_loop = partial(
        training_loop,
        loss_fn=loss_fn,
        epochs=epochs,
        patience=patience,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=output_path,
    )
    best_model = train_loop(data=loader, model=model)
    with torch.no_grad():
        l1 = torch.nn.L1Loss()
        wp_loss = l1(
            best_model(embeddings.word_embeddings.weight), 
            embeddings.word_embeddings.weight
        )
        pos_loss = l1(
            best_model(embeddings.position_embeddings.weight), 
            embeddings.position_embeddings.weight
        )
        typ_loss = l1(
            best_model(embeddings.token_type_embeddings.weight), 
            embeddings.token_type_embeddings.weight
        )
        print(f"Word-piece loss {wp_loss}")
        print(f"Positional loss {pos_loss}")
        print(f"Token-type loss {typ_loss}")
        new_pos = array2embedding(
            best_model.encode(embeddings.position_embeddings.weight)
        )
        new_wp = array2embedding(
            best_model.encode(embeddings.word_embeddings.weight)
        )
        new_typ = array2embedding(
            best_model.encode(embeddings.token_type_embeddings.weight)
        )
    embeddings.word_embeddings = new_wp
    embeddings.position_embeddings = new_pos
    embeddings.token_type_embeddings = new_typ
    embeddings.projection = best_model.decoder
    embeddings.layer_norm = best_model.layer_norm
    nlp.config["components"]["transformer"]["model"]["embedding_size"] = model.compressed_size
    nlp.to_disk(output_path)


@app.command()
def noise_transformer(
    model_path: str,
    output_path: str
):
    nlp = spacy.load(model_path)
    trf_pipe = nlp.get_pipe("transformer")
    trf_model = trf_pipe.model.get_ref("transformer")
    trf_pytorch = trf_model.shims[0]._model
    embeddings = trf_pytorch.embeddings.inner
    wp = torch.rand(embeddings.word_embeddings.weight.shape) / 10
    pos = torch.rand(embeddings.position_embeddings.weight.shape) / 10
    typ = torch.rand(embeddings.token_type_embeddings.weight.shape) / 10
    loss = torch.nn.L1Loss()
    print(
        loss(
            embeddings.word_embeddings.weight,
            embeddings.word_embeddings.weight + wp
        )
    )
    wp += embeddings.word_embeddings.weight
    pos += embeddings.position_embeddings.weight
    typ += embeddings.token_type_embeddings.weight
    embeddings.word_embeddings = array2embedding(wp)
    embeddings.position_embeddings = array2embedding(pos)
    embeddings.token_type_embeddings = array2embedding(typ)
    nlp.to_disk(output_path)
    


if __name__ == "__main__":
    app()
