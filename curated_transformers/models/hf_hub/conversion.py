from typing import Any, Callable, Dict, List, Mapping, Tuple, Union

from torch import Tensor

from ...util.string import StringTransform, StringTransformations


def process_hf_keys(
    model_name: str,
    hf_config: Dict[str, Any],
    hf_to_curated: Dict[str, Union[str, Tuple[str, Callable]]],
    extra_keys: List[str] = [],
) -> Dict[str, Any]:
    """
    Convert Hugging Face configuration keys to keyword arguments for
    Curated Transformers configuration classes.

    :param model_name:
        Model name. Only used in exception messages.
    :param hf_config:
        Hugging Face model configuration.
    :param hf_to_curated:
        Dictionary that maps Hugging Face configuration keys to keyword
        arguments for a Curated Transformers configuration class. If a value
        is a tuple, the first tuple element is the name of the keyword
        argument class and the second tuple element is a conversion function.
    :param extra_keys:
        Optional keys for which the Hugging Face configuration key and the
        keyword argument of the Curated Transformers configuration class is
        the same.
    :returns:
        Dictionary with keyword arguments.
    """
    missing_keys = tuple(
        sorted(set(hf_to_curated.keys()).difference(set(hf_config.keys())))
    )
    if len(missing_keys) != 0:
        raise ValueError(
            f"Missing keys in Hugging Face {model_name} model config: {missing_keys}"
        )

    kwargs = {}

    for hf, curated in hf_to_curated.items():
        if isinstance(curated, tuple):
            curated, ctor = curated
        else:
            ctor = lambda x: x

        kwargs[curated] = ctor(hf_config[hf])

    # Handle config options that are not set in all models.
    kwargs.update({k: hf_config[k] for k in extra_keys if k in hf_config})

    return kwargs


def state_dict_from_hf(
    params: Mapping[str, Tensor], transforms: List[StringTransform]
) -> Mapping[str, Tensor]:
    """
    Apply transformations to a Hugging Face state dict to make it
    compatible with Curated Transformer modules.

    :param params:
        Hugging Face state dict.
    :param transforms:
        List of string transformations for the state dict's keys.
    :returns:
        Transformed state dict.
    """
    out = {}
    for key, param in params.items():
        for transform in transforms:
            key = transform.apply(key)
        out[key] = param
    return out


def state_dict_to_hf(
    params: Mapping[str, Tensor], transforms: List[StringTransform]
) -> Mapping[str, Tensor]:
    """
    Apply transformations to a Curated Transformer state dict to make it
    compatible with Hugging Face modules.

    :param params:
        Curated Transformer state dict.
    :param transforms:
        List of string transformations for the state dict's keys.
        This must be the same transformations that were used to
        convert the original Hugging Face state dict.
    :returns:
        Transformed state dict.
    """
    out = {}
    for key, param in params.items():
        for transform in transforms[::-1]:
            key = transform.revert(key)
        out[key] = param
    return out
