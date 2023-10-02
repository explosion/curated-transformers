from enum import Enum
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

import torch

from ..._compat import has_safetensors
from ...repository.file import RepositoryFile

if TYPE_CHECKING:
    import safetensors


class ModelCheckpointType(Enum):
    """
    Types of model checkpoints supported by Curated Transformers.
    """

    #: PyTorch `checkpoint<https://pytorch.org/docs/stable/generated/torch.save.html>`_.
    PYTORCH_STATE_DICT = 0

    #: Hugging Face `Safetensors <https://github.com/huggingface/safetensors>`_ checkpoint.
    SAFE_TENSORS = 1

    @property
    def loader(
        self,
    ) -> Callable[[Iterable[RepositoryFile]], Iterable[Mapping[str, torch.Tensor]]]:
        checkpoint_type_to_loader = {
            ModelCheckpointType.PYTORCH_STATE_DICT: _load_pytorch_state_dicts_from_checkpoints,
            ModelCheckpointType.SAFE_TENSORS: _load_safetensor_state_dicts_from_checkpoints,
        }
        return checkpoint_type_to_loader[self]

    @property
    def pretty_name(self) -> str:
        if self == ModelCheckpointType.PYTORCH_STATE_DICT:
            return "PyTorch StateDict"
        elif self == ModelCheckpointType.SAFE_TENSORS:
            return "SafeTensors"
        else:
            return ""


def _load_safetensor_state_dicts_from_checkpoints(
    checkpoints: Iterable[RepositoryFile],
) -> Iterable[Mapping[str, torch.Tensor]]:
    if not has_safetensors:
        raise ValueError(
            "The `safetensors` library is required to load models from Safetensors checkpoints"
        )

    import safetensors.torch

    for checkpoint in checkpoints:
        # Prefer to load from a path when possible. Since loading from a file
        # temporarily puts the checkpoint in memory twice.
        if checkpoint.path is not None:
            # Map to CPU first to support all devices.
            state_dict = safetensors.torch.load_file(checkpoint.path, device="cpu")
        else:
            with checkpoint.open() as f:
                # This has memory overhead, since Safetensors does not have
                # support for loading from a file object and cannot use
                # the bytes in-place.
                checkpoint_bytes = f.read()
                state_dict = safetensors.torch.load(checkpoint_bytes)
        yield state_dict


def _load_pytorch_state_dicts_from_checkpoints(
    checkpoints: Iterable[RepositoryFile],
) -> Iterable[Mapping[str, torch.Tensor]]:
    for checkpoint in checkpoints:
        with checkpoint.open() as f:
            # Map to CPU first to support all devices.
            state_dict = torch.load(
                f, map_location=torch.device("cpu"), weights_only=True
            )
        yield state_dict
