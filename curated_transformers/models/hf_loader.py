from typing import Callable, List, Optional
from spacy.tokens import Doc

from .._compat import transformers, has_hf_transformers
from .pytorch.hf_util import convert_hf_pretrained_model_parameters
from .types import TorchTransformerModelT


def build_hf_encoder_loader_v1(
    *,
    name: str,
    revision: str = "main",
) -> Callable[
    [TorchTransformerModelT, Optional[List[Doc]], Optional[List[Doc]]],
    TorchTransformerModelT,
]:
    """Construct a callback that initializes a supported transformer
    model with weights from a corresponding HuggingFace model.

    name (str):
        Name of the HuggingFace model.
    revision (str):
        Name of the model revision/branch.
    """

    def load(model, X=None, Y=None):
        if not has_hf_transformers:
            raise ValueError("requires huggingface transformers")

        encoder = model.shims[0]._model

        hf_model = transformers.AutoModel.from_pretrained(name, revision=revision)
        params = convert_hf_pretrained_model_parameters(hf_model)
        encoder.load_state_dict(params)

        return model

    return load
