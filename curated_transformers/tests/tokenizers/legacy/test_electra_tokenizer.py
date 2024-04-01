import pytest

from curated_transformers.tokenizers.legacy.bert_tokenizer import BERTTokenizer

from ...compat import has_hf_transformers
from ..util import compare_tokenizer_outputs_with_hf_tokenizer


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize(
    "model_name",
    [
        "jonfd/electra-small-nordic",
        "Maltehb/aelaectra-danish-electra-small-cased",
        "google/electra-small-discriminator",
    ],
)
def test_from_hf_hub_equals_hf_tokenizer(model_name: str, sample_texts):
    compare_tokenizer_outputs_with_hf_tokenizer(sample_texts, model_name, BERTTokenizer)
