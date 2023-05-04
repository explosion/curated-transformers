from dataclasses import dataclass

from ..bert import BertConfig


@dataclass
class RobertaConfig(BertConfig):
    def __init__(
        self,
        *args,
        layer_norm_eps=1e-05,
        max_position_embeddings=514,
        padding_idx=1,
        type_vocab_size=1,
        vocab_size=50265,
        **kwargs
    ):
        super(RobertaConfig, self).__init__(
            *args,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            padding_idx=padding_idx,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size,
            **kwargs
        )
