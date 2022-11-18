import typer
import srsly
import spacy
import torch
import numpy as np

from spacy.util import ensure_path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from curated_transformers.models.roberta.embeddings import RobertaEmbeddings
from curated_transformers.models.bert.embeddings import BertEmbeddings
from curated_transformers.models.bert.config import BertEmbeddingConfig, BertLayerConfig
from model import make_autoencoder, make_twinembeddings
from data import make_transformer_loader, array2embedding, arrays2linear, arrays2layernorm
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
    out_path: str
):
    config = srsly.read_yaml(config_path)
    model_type = config["model"]["type"]
    if model_type != "autoencoder":
        raise NotImplementedError(
            "Transformer embedding compression, currently"
            "only works with the autoencoder"
        )
    if "huggingface" in config["loader"]:
        transformer = config["loader"]["hf-transformer"]
        source = "hf"
    elif "curated-transformers" in config["loader"]:
        transformer = config["loader"]["curated-transformers"]
        source = "curated"
    else:
        raise NotImplementedError(
            "Can only load transformer from curated-transformers "
            "and hf-transformers. Please put one of these strings "
            "as keys in the 'loader' section."
        )

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


@app.command()
def stitch_curated_transformer(
    model_path: str,
    embedding_path: str,
    output_path: str
):
    model_path = ensure_path(model_path)
    embedding_path = ensure_path(embedding_path)
    output_path = ensure_path(output_path)
    nlp = spacy.load(model_path)
    word_embeddings = array2embedding(
        np.load(embedding_path / "word_pieces.npy") 
    )
    token_type_embeddings = array2embedding(
        np.load(embedding_path / "token_type.npy")
    )
    position_embeddings = array2embedding(
        np.load(embedding_path / "positional.npy")
    )
    projection = arrays2linear(
        np.load(embedding_path / "weights.npy"),
        np.load(embedding_path / "bias.npy"),
    )
    layer_norm = arrays2layernorm(
        np.load(embedding_path / "ln_weight.npy"),
        np.load(embedding_path / "ln_bias.npy")
    )
    embedding_config = BertEmbeddingConfig(
        embedding_dim=word_embeddings.embedding_dim,
        vocab_size=word_embeddings.num_embeddings,
        type_vocab_size=token_type_embeddings.num_embeddings,
        max_position_embeddings=position_embeddings.num_embeddings
    )
    layer_config = BertLayerConfig()
    new_embeddings = BertEmbeddings(embedding_config, layer_config)
    new_embeddings.word_embeddings = word_embeddings
    new_embeddings.position_embeddings = position_embeddings
    new_embeddings.token_type_embeddings = token_type_embeddings
    new_embeddings.projection = projection
    trf_pipe = nlp.get_pipe("transformer")
    trf_model = trf_pipe.model.get_ref("transformer")
    trf_pytorch = trf_model.shims[0]._model
    if isinstance(trf_pytorch.embeddings, RobertaEmbeddings):
        padding_idx = trf_pytorch.embeddings.padding_idx
        roberta_embeddings = RobertaEmbeddings(
            embedding_config, layer_config, padding_idx=padding_idx
        )
        roberta_embeddings.inner = new_embeddings
        trf_pytorch.embeddings = roberta_embeddings
    else:
        trf_pytorch.embeddings = new_embeddings
    nlp.to_disk(output_path)


@app.command()
def compress_spacy(
    config_path,
    out_path
):
    print("future command for compressing spacy vectors.")


@app.command()
def stitch_spacy(
    model_path: str,
    embedding_path: str,
    output_path: str
):
    ...


@app.command()
def compress_vectors(
    config_path,
    out_path
):
    print("Future command for compressing other vectors.")




if __name__ == "__main__":
    app()
