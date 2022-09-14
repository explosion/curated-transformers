from dataclasses import dataclass


@dataclass
class BertConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    attention_probs_dropout_prob: float
    hidden_dropout_prob: float
    hidden_act: str
    vocab_size: int
    type_vocab_size: int
    max_position_embeddings: int
    model_max_length: int
    layer_norm_eps: float
    padding_idx: int

    def __init__(self):
        super(BertConfig, self).__init__()

        # From syntaxdot
        # https://github.com/tensordot/syntaxdot/blob/22bd3d43ed2d7fcbef8a6217b01684194fae713f/syntaxdot-transformers/src/models/bert/config.rs#L25
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.hidden_act = "gelu"
        self.vocab_size = 30000
        self.type_vocab_size = 2
        self.max_position_embeddings = 512
        self.model_max_length = 512
        self.layer_norm_eps = 1e-12
        self.padding_idx = 0
