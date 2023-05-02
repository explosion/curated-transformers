import pytest
from spacy.util import registry as spacy_registry
from curated_transformers.util import registry


@pytest.mark.parametrize(
    "encoder_name",
    [
        "curated-transformers.BertWordpieceEncoder.v1",
        "curated-transformers.ByteBpeEncoder.v1",
        "curated-transformers.CamembertSentencepieceEncoder.v1",
        "curated-transformers.CharEncoder.v1",
        "curated-transformers.SentencepieceEncoder.v1",
        "curated-transformers.WordpieceEncoder.v1",
        "curated-transformers.XlmrSentencepieceEncoder.v1",
    ],
)
def test_encoder_from_registry(encoder_name):
    spacy_registry.architectures.get(encoder_name)()


@pytest.mark.parametrize(
    "loader_name",
    [
        "curated-transformers.ByteBpeLoader.v1",
        "curated-transformers.CharEncoderLoader.v1",
        "curated-transformers.HFTransformerEncoderLoader.v1",
        "curated-transformers.HFPieceEncoderLoader.v1",
        "curated-transformers.PyTorchCheckpointLoader.v1",
        "curated-transformers.SentencepieceLoader.v1",
        "curated-transformers.WordpieceLoader.v1",
    ],
)
def test_encoder_loader_from_registry(loader_name):
    # Can't be constructed, since most loaders have mandatory arguments.
    registry.model_loaders.get(loader_name)
