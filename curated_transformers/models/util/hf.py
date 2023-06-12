import itertools
from typing import List, Mapping
import torch
import re

from .serde import DeserializationParamBucket, RegExParameterBucket


def _rename_old_hf_names(
    params: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    out = {}
    for name, parameter in params.items():
        name = re.sub(r"\.gamma$", ".weight", name)
        name = re.sub(r"\.beta$", ".bias", name)
        out[name] = parameter
    return out


def _param_buckets_for_bert_qkv(
    num_layers: int,
) -> List[DeserializationParamBucket]:
    out = []
    for layer in range(num_layers):
        # This has to match the parameter key **BEFORE** it's renamed, i.e.,
        # the key used in the original pre-trained checkpoint from HF Hub.
        regex_str = f"\\.{layer}\\.attention\\.self\\.(query|key|value)\\.(weight|bias)"
        expected_keys = {
            f".{layer}.attention.self.{module}.{param}"
            for module, param in itertools.product(
                ["query", "key", "value"], ["weight", "bias"]
            )
        }
        out.append(RegExParameterBucket(pattern=regex_str, expected_keys=expected_keys))
    return out  # type: ignore


def _merge_qkv(params: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    out = {}
    for name, parameter in params.items():
        # These regexps correspond to the parameter keys **AFTER** normalization.
        m = re.match(
            r"layers\.(?P<layer>[0-9]+)\.mha\.(query|key|value).(?P<param_type>weight|bias)",
            name,
        )
        if m:
            if "query" in name:
                base = f"layers.{m['layer']}.mha"
                out[f"{base}.input.{m['param_type']}"] = torch.cat(
                    [
                        parameter,
                        params[f"{base}.key.{m['param_type']}"],
                        params[f"{base}.value.{m['param_type']}"],
                    ]
                )
            continue
        out[name] = parameter

    return out
