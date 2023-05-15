from typing import Mapping
import torch
import re


def _rename_old_hf_names(
    params: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    out = {}
    for name, parameter in params.items():
        name = re.sub(r"\.gamma$", ".weight", name)
        name = re.sub(r"\.beta$", ".bias", name)
        out[name] = parameter
    return out


def _merge_qkv(params: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    out = {}
    for name, parameter in params.items():
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
