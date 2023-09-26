from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Optional

import torch

from .._compat import has_safetensors
from ..repository.file import RepositoryFile

if TYPE_CHECKING:
    import safetensors


HF_MODEL_CONFIG = "config.json"
HF_MODEL_CHECKPOINT = "pytorch_model.bin"
HF_MODEL_CHECKPOINT_SAFETENSORS = "model.safetensors"
HF_MODEL_SHARDED_CHECKPOINT_INDEX = "pytorch_model.bin.index.json"
HF_MODEL_SHARDED_CHECKPOINT_INDEX_SAFETENSORS = "model.safetensors.index.json"
HF_MODEL_SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY = "weight_map"
HF_TOKENIZER_CONFIG = "tokenizer_config.json"
SPECIAL_TOKENS_MAP = "special_tokens_map.json"
TOKENIZER_JSON = "tokenizer.json"


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


PRIMARY_CHECKPOINT_FILENAMES = {
    ModelCheckpointType.PYTORCH_STATE_DICT: HF_MODEL_CHECKPOINT,
    ModelCheckpointType.SAFE_TENSORS: HF_MODEL_CHECKPOINT_SAFETENSORS,
}
SHARDED_CHECKPOINT_INDEX_FILENAMES = {
    ModelCheckpointType.PYTORCH_STATE_DICT: HF_MODEL_SHARDED_CHECKPOINT_INDEX,
    ModelCheckpointType.SAFE_TENSORS: HF_MODEL_SHARDED_CHECKPOINT_INDEX_SAFETENSORS,
}
# Same for both checkpoint types.
SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY = HF_MODEL_SHARDED_CHECKPOINT_INDEX_WEIGHTS_KEY


# When `None`, behaviour is implementation-specific.
_MODEL_CHECKPOINT_TYPE: ContextVar[Optional[ModelCheckpointType]] = ContextVar(
    "model_checkpoint_type", default=None
)


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
