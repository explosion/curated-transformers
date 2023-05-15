import pytest
import torch

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.models.bert.encoder import BertEncoder

from ...util import torch_assertclose


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.slow
def test_encoder():
    hf_model = transformers.AutoModel.from_pretrained("bert-base-cased")
    hf_model.eval()

    model = BertEncoder.from_hf_hub("bert-base-cased")
    model.eval()

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10))

    with torch.no_grad():
        Y = model(X).last_hidden_layer_states
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf)
