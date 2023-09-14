import pytest

from curated_transformers.tokenizers.legacy import LlamaTokenizer

from ...compat import has_hf_transformers
from ..util import compare_tokenizer_outputs_with_hf_tokenizer


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_from_hf_hub_equals_hf_tokenizer(sample_texts):
    # OpenLlama does not provide a fast tokenizer. If we ask for a fast
    # tokenizer, the slow tokenizer gets converted, which takes too much
    # time for CI. So, we use the slow tokenizer instead.
    compare_tokenizer_outputs_with_hf_tokenizer(
        sample_texts,
        "openlm-research/open_llama_3b",
        LlamaTokenizer,
        with_hf_fast=False,
        pad_token="<unk>",
    )
