import re
from typing import Mapping

import torch


def _rename_old_hf_names(
    params: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    out = {}
    for name, parameter in params.items():
        name = re.sub(r"\.gamma$", ".weight", name)
        name = re.sub(r"\.beta$", ".bias", name)
        out[name] = parameter
    return out
