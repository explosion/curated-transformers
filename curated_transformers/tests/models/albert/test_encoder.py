import pytest
import torch

from curated_transformers._compat import has_hf_transformers, transformers
from curated_transformers.models.albert import AlbertConfig, AlbertEncoder


from ...util import torch_assertclose


def test_rejects_incorrect_number_of_groups():
    config = AlbertConfig(num_hidden_groups=5)
    with pytest.raises(ValueError, match=r"must be divisable"):
        AlbertEncoder(config)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.slow
def test_encoder():
    hf_model = transformers.AutoModel.from_pretrained("albert-base-v2")
    hf_model.eval()

    model = AlbertEncoder.from_hf_hub("albert-base-v2")
    model.eval()

    X = torch.randint(0, hf_model.config.vocab_size, (2, 10))

    with torch.no_grad():
        Y = model(X).last_hidden_layer_states
        Y_hf = hf_model(X).last_hidden_state

    torch_assertclose(Y, Y_hf)
