import pytest

from spacy_experimental.transformers.models import TransformerEncoder
from spacy_experimental.transformers.models.util import convert_hf_pretrained_model_parameters

@pytest.mark.slow
def test_hf_load_roberta_weights():
    from transformers import RobertaModel, RobertaConfig
    
    hf_model = RobertaModel.from_pretrained("roberta-base")
    params = convert_hf_pretrained_model_parameters(hf_model)

    config: RobertaConfig = hf_model.config
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
