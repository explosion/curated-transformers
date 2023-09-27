from typing import Any, Dict, Mapping

from ..hf_hub.conversion import HFSpecificConfig
from ..roberta._hf import _config_from_hf as _roberta_config_from_hf
from ..roberta._hf import _config_to_hf as _roberta_config_to_hf
from ..roberta.config import RoBERTaConfig

HF_SPECIFIC_CONFIG = HFSpecificConfig(
    architectures=["CamembertModel"], model_type="camembert"
)


def _config_from_hf(hf_config: Mapping[str, Any]) -> RoBERTaConfig:
    return _roberta_config_from_hf(hf_config)


def _config_to_hf(curated_config: RoBERTaConfig) -> Dict[str, Any]:
    out = _roberta_config_to_hf(curated_config)
    return HF_SPECIFIC_CONFIG.merge(out)
