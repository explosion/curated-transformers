import pytest

from curated_transformers.generation import AutoGenerator
from curated_transformers.generation.dolly_v2 import DollyV2Generator
from curated_transformers.generation.falcon import FalconGenerator
from curated_transformers.generation.llama import LlamaGenerator
from curated_transformers.generation.mpt import MPTGenerator


@pytest.mark.slow
def test_auto_generator():
    model_causallm_map = {
        "databricks/dolly-v2-3b": DollyV2Generator,
        "tiiuae/falcon-7b": FalconGenerator,
        "openlm-research/open_llama_3b": LlamaGenerator,
        "mosaicml/mpt-7b": MPTGenerator,
    }

    for name, generator_cls in model_causallm_map.items():
        generator = AutoGenerator.from_hf_hub(name=name)
        assert isinstance(generator, generator_cls)

    with pytest.raises(ValueError, match="Unsupported generator"):
        AutoGenerator.from_hf_hub(name="trl-internal-testing/tiny-random-GPT2Model")
