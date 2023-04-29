from typing import Optional
from io import BytesIO
from pathlib import Path
import spacy
from spacy import Language
from spacy.cli import app
from thinc.api import Model, PyTorchShim, pytorch_to_torchscript_wrapper
import torch
import torch.nn.quantized as nnq
from torch.nn import Embedding, Linear, Module, MSELoss
from torch.quantization import qconfig
from typer import Argument as Arg, Option
import warnings

from ..errors import Warnings, Errors
from ..pipeline.transformer import Transformer


MODULE_QUANTIZERS = {
    Embedding: qconfig.float_qparams_weight_only_qconfig,
    Linear: qconfig.default_dynamic_qconfig,
}


@app.command("quantize-transformer")
def quantize_cli(
    model_path: Path = Arg(..., help="Model to quantize", exists=True, allow_dash=True),
    output_path: Path = Arg(..., help="Output directory to store quantized model in"),
    max_mse_loss: Optional[float] = Option(
        None, "--max-mse-loss", help="Maximum MSE loss of quantized parameters"
    ),
    skip_embeds: bool = Option(
        False, "--skip-embeds", help="Do not quantize embeddings"
    ),
    skip_linear: bool = Option(
        False, "--skip-linear", help="Do not quantize linear layers"
    ),
):
    """
    Quantize a curated transformers model to reduce its size.
    """
    nlp = spacy.load(model_path)
    nlp_quantize_dynamic(
        nlp, max_mse_loss=max_mse_loss, skip_embeds=skip_embeds, skip_linear=skip_linear
    )
    nlp.to_disk(output_path)


def size_of_model(model: Module):
    filelike = BytesIO()
    torch.save(model.state_dict(), filelike)
    return filelike.tell()


def nlp_quantize_dynamic(
    nlp: Language,
    *,
    max_mse_loss: Optional[float] = None,
    skip_embeds: bool = False,
    skip_linear: bool = False,
):
    for pipe_name, pipe in nlp.components:
        # We only quantize curated transformers, other models may not
        # be able to deal with quantization and/or TorchScript.
        if not isinstance(pipe, Transformer):
            continue

        model = pipe.model.get_ref("transformer")

        if model.name == "pytorch_script":  # Probably already quantized.
            warnings.warn(Warnings.W001)
            continue

        before_size = size_of_model(pytorch_model(model))
        quantize_dynamic(
            model,
            max_mse_loss=max_mse_loss,
            skip_embeds=skip_embeds,
            skip_linear=skip_linear,
        )
        after_size = size_of_model(pytorch_model(model))
        print(
            f"Quantized model in pipe '{pipe_name}' ({before_size/2**20:.1f} MiB -> {after_size/2**20:.1f} MiB)"
        )

        pipe.model.replace_node(model, pytorch_to_torchscript_wrapper(model))
        nlp.config["components"][pipe_name]["model"]["torchscript"] = True


def quantize_dynamic(
    model: Model,
    max_mse_loss: Optional[float] = None,
    skip_embeds: bool = False,
    skip_linear: bool = False,
):
    assert model.name == "pytorch"
    assert isinstance(model.shims[0], PyTorchShim)

    pytorch_model = model.shims[0]._model
    quantize_types = {}

    if not skip_embeds:
        quantize_types[Embedding] = MODULE_QUANTIZERS[Embedding]

    if not skip_linear:
        quantize_types[Linear] = MODULE_QUANTIZERS[Linear]  # type: ignore

    quantized_model = torch.quantization.quantize_dynamic(
        pytorch_model,
        quantize_types,
        dtype=torch.qint8,
    )

    if max_mse_loss:
        quantized_model = requantize_with_max_loss(
            pytorch_model, quantized_model, max_mse_loss
        )

    model.shims[0]._model = quantized_model


def requantize_with_max_loss(
    model: Module, quantized_model: Module, max_mse_loss: float
):
    loss = MSELoss()

    modules = {name: child for name, child in model.named_modules()}
    skipped_modules = []
    module_qconfig = {}
    for name, qmodule in quantized_model.named_modules():
        if not (isinstance(qmodule, nnq.Embedding) or isinstance(qmodule, nnq.Linear)):
            continue

        module = modules[name]
        mse_loss = loss(torch.dequantize(qmodule.weight()), module.weight)
        if mse_loss <= max_mse_loss:
            module_qconfig[name] = MODULE_QUANTIZERS[type(module)]
        else:
            skipped_modules.append(name)

    print(f"Skipping modules with MSE > {max_mse_loss}: {', '.join(skipped_modules)}")

    return torch.quantization.quantize_dynamic(
        model,
        module_qconfig,
        dtype=torch.qint8,
    )


def pytorch_model(model: Model) -> Module:
    if model.name != "pytorch" or not isinstance(model.shims[0], PyTorchShim):
        raise ValueError(Errors.E001.format(model_name=model.name))

    return model.shims[0]._model
