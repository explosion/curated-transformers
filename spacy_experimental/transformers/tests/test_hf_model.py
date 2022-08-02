import pytest

# fmt: off
from spacy_experimental.transformers.models import TransformerEncoder
from spacy_experimental.transformers.models.util import convert_hf_pretrained_model_parameters
from spacy_experimental.transformers.models.util import has_hf_transformers, transformers
# fmt: on


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
@pytest.mark.parametrize("model_name", ["roberta-base", "xlm-roberta-base"])
def test_hf_load_roberta_weights(model_name):
    from transformers import AutoModel, AutoConfig

    hf_model = AutoModel.from_pretrained(model_name)
    params = convert_hf_pretrained_model_parameters(hf_model)

    config: AutoConfig = hf_model.config
    encoder = TransformerEncoder(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        n_heads=config.num_attention_heads,
        n_layers=config.num_hidden_layers,
        attn_dropout=config.attention_probs_dropout_prob,
        hidden_dropout=config.hidden_dropout_prob,
        hidden_activation=config.hidden_act,
        max_pos_embeddings=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        learnable_pos_embeddings=True,
        padding_idx=config.pad_token_id,
    )

    encoder.load_state_dict(params)
