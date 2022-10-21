from typing import List
from spacy.tokens import Doc
from thinc.api import Model

from .._compat import transformers, has_hf_transformers
from .hf_util import convert_hf_pretrained_model_parameters


def build_hf_encoder_loader_v1(
    *,
    name: str,
    revision: str = "main",
):
    def load(model: Model, X: List[Doc] = None, Y=None):
        if not has_hf_transformers:
            raise ValueError("requires 🤗 transformers")

        global transformers
        from transformers import AutoModel

        encoder = model.shims[0]._model

        hf_model = AutoModel.from_pretrained(name, revision=revision)
        params = convert_hf_pretrained_model_parameters(hf_model)
        encoder.load_state_dict(params)

        return model

    return load
