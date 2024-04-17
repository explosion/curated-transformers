import pytest
from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

from curated_transformers.tokenizers import Tokenizer
from curated_transformers.tokenizers.legacy import BERTTokenizer

from ..compat import has_hf_transformers
from .util import compare_tokenizer_outputs_with_hf_tokenizer


def test_from_hf_hub_to_cache():
    Tokenizer.from_hf_hub_to_cache(
        name="EleutherAI/gpt-neox-20b",
        revision="9369f145ca7b66ef62760f9351af951b2d53b77f",
    )

    expected_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
    ]
    for name in expected_files:
        assert (
            try_to_load_from_cache(
                repo_id="EleutherAI/gpt-neox-20b",
                filename=name,
                revision="9369f145ca7b66ef62760f9351af951b2d53b77f",
            )
            != _CACHED_NO_EXIST
        )


def test_from_hf_hub_to_cache_legacy():
    BERTTokenizer.from_hf_hub_to_cache(
        name="bert-base-uncased",
        revision="1dbc166cf8765166998eff31ade2eb64c8a40076",
    )

    expected_files = [
        "tokenizer_config.json",
        "vocab.txt",
    ]
    for name in expected_files:
        assert (
            try_to_load_from_cache(
                repo_id="bert-base-uncased",
                filename=name,
                revision="1dbc166cf8765166998eff31ade2eb64c8a40076",
            )
            != _CACHED_NO_EXIST
        )


@pytest.mark.xfail(reason="HfFileSystem calls safetensors with incorrect arguments")
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_fsspec(sample_texts):
    # We only test one model, since using fsspec downloads the model
    # each time.
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts,
        "EleutherAI/gpt-neox-20b",
        Tokenizer,
        pad_token="</s>",
        with_fsspec=True,
    )
