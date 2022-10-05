from io import BytesIO
from pathlib import Path
import spacy
from spacy import Language
from spacy.cli import app
from thinc.api import Model, PyTorchShim
import torch
from torch.nn import Embedding, Linear, Module
from torch.quantization import qconfig
from typer import Argument as Arg, Option

from ..models.torchscript_wrapper import to_torchscript_wrapper
from ..pipe import Transformer


@app.command("quantize-transformer")
def quantize_cli(
    model_path: Path = Arg(..., help="Model to quantize", exists=True, allow_dash=True),
    output_path: Path = Arg(..., help="Output directory to store quantized model in"),
    skip_embeds: bool = Option(
        False, "--skip-embeds", help="Do not quantize embeddings"
    ),
    skip_linear: bool = Option(
        False, "--skip-linear", help="Do not quantize linear layers"
    ),
):
    """
    Quantize a curated-transformers model.
    """
    nlp = spacy.load(model_path)
    nlp_quantize_dynamic(nlp, skip_embeds=skip_embeds, skip_linear=skip_linear)
    nlp.to_disk(output_path)


def size_of_model(model):
    filelike = BytesIO()
    torch.save(model.state_dict(), filelike)
    return filelike.tell()


def nlp_quantize_dynamic(
    nlp: Language, *, skip_embeds: bool = False, skip_linear: bool = False
):
    for pipe_name, pipe in nlp.components:
        # We only quantize curated transformers, other models may not
        # be able to deal with quantization and/or TorchScript.
        if not isinstance(pipe, Transformer):
            continue

        model = pipe.model.get_ref("transformer")

        if model.name == "pytorch_script":  # Probably already quantized.
            continue

        before_size = size_of_model(pytorch_model(model))
        quantize_dynamic(model, skip_embeds=skip_embeds, skip_linear=skip_linear)
        after_size = size_of_model(pytorch_model(model))
        print(
            f"Quantized model in pipe '{pipe_name}' ({before_size/2**20:.1f} MiB -> {after_size/2**20:.1f} MiB)"
        )

        pipe.model.replace_node(model, to_torchscript_wrapper(model))
        nlp.config["components"][pipe_name]["model"]["torchscript"] = True


def quantize_dynamic(
    model: Model, skip_embeds: bool = False, skip_linear: bool = False
):
    assert model.name == "pytorch"
    assert isinstance(model.shims[0], PyTorchShim)

    pytorch_model = model.shims[0]._model
    quantize_types = {}

    if not skip_linear:
        quantize_types[Linear] = qconfig.default_dynamic_qconfig

    if not skip_embeds:
        quantize_types[Embedding] = qconfig.float_qparams_weight_only_qconfig

    pytorch_model = torch.quantization.quantize_dynamic(
        pytorch_model,
        quantize_types,
        dtype=torch.qint8,
    )

    model.shims[0]._model = pytorch_model


def pytorch_model(model: Model) -> Module:
    if model.name != "pytorch":
        raise ValueError("Cannot extract PyTorch model from f{model.name}")

    if not isinstance(model.shims[0], PyTorchShim):
        raise ValueError("Model does not hold a PyTorchShim")

    return model.shims[0]._model
