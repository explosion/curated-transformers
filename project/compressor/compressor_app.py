import srsly
from model import make_autoencoder, make_twinembeddings
from data import make_loader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_model(config):
    model_config = config["model"]
    original_size = model_config["original_size"]
    ratio = model_config["ratio"]
    compress_size = original_size / ratio
    if model_config["type"] == "autoencoder":
        model = make_autoencoder(
            activation=model_config["activation"],
            in_dim=original_size,
            out_dim=compress_size,
            normalize=model_config["normalize"],
            hidden_dims=model_config["hidden_dims"],
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


def compress(
    data_path: str,
    config_path: str,
    in_dim: int,
    ratio: int
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
    optimizer = AdamW(model.parameters, learning_rate, weight_decay)
    factor = config["scheduler"]["factor"]
    patience = config["sheduler"]["patience"]
    scheduler = ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    loss_fn = 
