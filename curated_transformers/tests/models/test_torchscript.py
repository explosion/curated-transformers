import copy
import pytest
import torch.jit

from curated_transformers.models.albert.config import AlbertConfig
from curated_transformers.models.albert.encoder import AlbertEncoder
from curated_transformers.models.bert.config import BertConfig
from curated_transformers.models.bert.encoder import BertEncoder
from curated_transformers.models.roberta.config import RobertaConfig
from curated_transformers.models.roberta.encoder import RobertaEncoder
from curated_transformers.models.curated_transformer import CuratedTransformer


ENCODER_FACTORIES = [
    (AlbertEncoder, AlbertConfig),
    (BertEncoder, BertConfig),
    (RobertaEncoder, RobertaConfig),
]


@pytest.mark.slow
@pytest.mark.parametrize("factories", ENCODER_FACTORIES)
def test_encoder_deepcopy(factories):
    # Not necessarily a TorchScript test, but we often want to
    # copy a module before TorchScript conversion (see e.g.
    # quantization).

    # Use a small vocab to limit memory use.
    encoder_factory, config_factory = factories
    encoder = CuratedTransformer(encoder_factory(config_factory(vocab_size=128)))
    copy.deepcopy(encoder)


@pytest.mark.slow
@pytest.mark.parametrize("factories", ENCODER_FACTORIES)
def test_encoder_torchscript(factories):
    # Use a small vocab to limit memory use.
    encoder_factory, config_factory = factories
    encoder = CuratedTransformer(encoder_factory(config_factory(vocab_size=128)))
    torch.jit.script(encoder)
