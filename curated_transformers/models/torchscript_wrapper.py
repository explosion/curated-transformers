from typing import Any, Callable, Optional
from io import BytesIO
import srsly  # type: ignore
from thinc.api import Model, PyTorchGradScaler, PyTorchShim, get_torch_default_device
from thinc.layers.pytorchwrapper import convert_pytorch_default_inputs
from thinc.layers.pytorchwrapper import convert_pytorch_default_outputs
from thinc.layers.pytorchwrapper import forward
import torch
from torch.jit import ScriptModule
from torch.nn import Module


class TorchScriptShim(PyTorchShim):
    """A Thinc shim that wraps a TorchScript module.

    model:
        The TorchScript module. A value of `None` is also possible to
        construct a shim to deserialize into.
    mixed_precision:
        Enable mixed-precision. This changes whitelisted ops to run
        in half precision for better performance and lower memory use.
    grad_scaler:
        The gradient scaler to use for mixed-precision training. If this
        argument is set to "None" and mixed precision is enabled, a gradient
        scaler with the default configuration is used.
    device:
        The PyTorch device to run the model on. When this argument is
        set to "None", the default device for the currently active Thinc
        ops is used.
    """

    def __init__(
        self,
        model: Optional[ScriptModule],
        config=None,
        optimizer: Any = None,
        mixed_precision: bool = False,
        grad_scaler: Optional[PyTorchGradScaler] = None,
        device: Optional["torch.device"] = None,
    ):
        if not isinstance(model, ScriptModule) and model is not None:
            raise ValueError(
                "PyTorchScriptShim must be initialized with ScriptModule or None (for deserialization)"
            )

        super().__init__(model, config, optimizer, mixed_precision, grad_scaler, device)

    def to_bytes(self) -> bytes:
        filelike = BytesIO()
        torch.jit.save(self._model, filelike)
        filelike.seek(0)
        model_bytes = filelike.getvalue()
        msg = {"config": self.cfg, "model": model_bytes}
        return srsly.msgpack_dumps(msg)

    def from_bytes(self, bytes_data: bytes) -> "TorchScriptShim":
        device = get_torch_default_device()
        msg = srsly.msgpack_loads(bytes_data)
        self.cfg = msg["config"]
        filelike = BytesIO(msg["model"])
        filelike.seek(0)
        self._model = torch.jit.load(filelike, map_location=device)
        self._model.to(device)
        self._grad_scaler.to_(device)
        return self


def TorchScriptWrapper_v1(
    model: Optional[ScriptModule] = None,
    convert_inputs: Optional[Callable] = None,
    convert_outputs: Optional[Callable] = None,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None,
    device: Optional["torch.device"] = None,
) -> Model[Any, Any]:
    """Wrap a TorchScript model, so that it has the same API as Thinc models.

    model:
        The TorchScript module. A value of `None` is also possible to
        construct a shim to deserialize into.
    mixed_precision:
        Enable mixed-precision. This changes whitelisted ops to run
        in half precision for better performance and lower memory use.
    grad_scaler:
        The gradient scaler to use for mixed-precision training. If this
        argument is set to "None" and mixed precision is enabled, a gradient
        scaler with the default configuration is used.
    device:
        The PyTorch device to run the model on. When this argument is
        set to "None", the default device for the currently active Thinc
        ops is used.
    """

    if convert_inputs is None:
        convert_inputs = convert_pytorch_default_inputs
    if convert_outputs is None:
        convert_outputs = convert_pytorch_default_outputs

    return Model(
        "pytorch_script",
        forward,
        attrs={"convert_inputs": convert_inputs, "convert_outputs": convert_outputs},
        shims=[
            TorchScriptShim(
                model=model,
                mixed_precision=mixed_precision,
                grad_scaler=grad_scaler,
                device=device,
            )
        ],
        dims={"nI": None, "nO": None},
    )


def to_torchscript_wrapper(model: Model) -> Model:
    """Convert a PyTorch wrapper to a TorchScript wrapper. The embedded PyTorch
    `Module` is converted to `ScriptModule`.
    """

    if model.name != "pytorch":
        raise ValueError(
            "Only PyTorch wrappers can be converted to TorchScript wrappers"
        )

    shim = model.shims[0]
    if not isinstance(shim, PyTorchShim):
        raise ValueError("Expected PyTorchShim when converting a PyTorch wrapper")

    convert_inputs = model.attrs["convert_inputs"]
    convert_outputs = model.attrs["convert_outputs"]

    pytorch_model = shim._model
    if not isinstance(pytorch_model, Module):
        raise ValueError("PyTorchShim does wrap a PyTorch module")

    torchscript_model = torch.jit.script(pytorch_model)
    grad_scaler = shim._grad_scaler
    mixed_precision = shim._mixed_precision
    device = shim.device

    return TorchScriptWrapper_v1(
        torchscript_model,
        convert_inputs=convert_inputs,
        convert_outputs=convert_outputs,
        mixed_precision=mixed_precision,
        grad_scaler=grad_scaler,
        device=device,
    )
