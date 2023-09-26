import pytest
from huggingface_hub import HfFileSystem

from curated_transformers.repository.fsspec import FsspecArgs
from curated_transformers.tokenizers import AutoTokenizer

_MODELS = [
    # tokenizer.json-only
    ("EleutherAI/gpt-neox-20b", "9369f145ca7b66ef62760f9351af951b2d53b77f"),
    # sentencepiece binary
    ("openlm-research/open_llama_13b", "b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8"),
    # tokenizer config, but no class name
    ("GroNLP/bert-base-dutch-cased", "484ff5cec2ad42b434537dadd901d9b8e2b64cd2"),
]


@pytest.mark.parametrize("model_revision", _MODELS)
def test_auto_tokenizer(model_revision):
    name, revision = model_revision
    AutoTokenizer.from_hf_hub(name=name, revision=revision)


@pytest.mark.slow
@pytest.mark.parametrize("model_revision", _MODELS)
def test_auto_tokenizer_fsspec(model_revision):
    name, revision = model_revision
    AutoTokenizer.from_fsspec(
        fs=HfFileSystem(), model_path=name, fsspec_args=FsspecArgs(revision=revision)
    )
    AutoTokenizer.from_hf_hub(name=name, revision=revision)


def test_cannot_infer():
    # This repo/revision does not have a tokenizer and doesn't match a
    # legacy tokenizer.
    with pytest.raises(ValueError, match=r"Cannot infer tokenizer for repo"):
        AutoTokenizer.from_hf_hub(
            name="explosion-testing/falcon-test",
            revision="235f4b64c489e33ae7e40163ab1f266bbe355651",
        )
