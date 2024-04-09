import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

from ...layers import Activation
from ...util.string import StringTransform
from ..config import TransformerConfig


@dataclass
class HFSpecificConfig:
    """
    Configuration options that must always be present in
    a Hugging Face compatible config file.

    The transformer version is the same for all of our
    models.

    :param architectures:
        Model-specific Hugging Face classes that can load the model.
    :param model_type:
        Hugging Face model type.
    :transformers_version:
        Minimum version of the ``transformers`` required to load the model.
    """

    architectures: List[str]
    model_type: str
    transformers_version: str = "4.33.0"

    def merge(self, hf_config: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Merges the keys with a Hugging Face configuration.

        :param hf_config:
            Hugging Face model configuration.
        :returns:
            Merged configuration.
        """
        out = dict(hf_config)
        out.update(dataclasses.asdict(self))
        return out


@dataclass
class HFConfigKey:
    """
    Metadata for a Hugging Face configuration key that describes
    how it must be (de-)serialized.

    :param name:
        Name of the key in the Hugging Face config file.
    :param mapping_to_curated_config:
        Either a string or a tuple of a string and a callable.
        The string is the name of the keyword argument passed
        to the constructor of the Curated Transformers config and
        the callable accepts the value of the key and returns a
        converted value.
    :param mapping_from_curated_config:
        A callable that accepts the config instance and returns
        a converted value.
    """

    name: str
    mapping_to_curated_config: Union[str, Tuple[str, Callable[[Any], Any]]]
    mapping_from_curated_config: Callable[[Any], Any]

    @property
    def curated_config_kwarg(self) -> str:
        """
        Returns the name of the Curated Transformer config keyword
        argument.
        """
        return (
            self.mapping_to_curated_config
            if isinstance(self.mapping_to_curated_config, str)
            else self.mapping_to_curated_config[0]
        )

    def get_kwarg(self, kwargs: Dict[str, Any]) -> Any:
        """
        Looks up the value of the key in the given keyword
        argument dictionary.

        :param kwargs:
            Keyword argument dictionary.
        :returns:
            Value of the key.
        """
        return kwargs[self.curated_config_kwarg]

    def set_kwarg(self, value: Union["HFConfigKey", Any], kwargs: Dict[str, Any]):
        """
        Converts and sets the value of the keyword argument in
        the given dictionary.

        :param value:
            Either a config key instance or a value. In the case
            of the former, its value is looked up in the keyword
            argument dictionary and saved as the calling config
            key's value as-is. In the case of the latter, the
            value is first converted (if possible) before being
            set.
        :param kwargs:
            Keyword argument dictionary.
        """
        if isinstance(value, HFConfigKey):
            kwargs[self.curated_config_kwarg] = value.get_kwarg(kwargs)
            return

        if isinstance(self.mapping_to_curated_config, tuple):
            curated, ctor = self.mapping_to_curated_config
        else:
            curated = self.mapping_to_curated_config
            ctor = lambda x: x

        kwargs[curated] = ctor(value)

    def remove_kwarg(self, kwargs: Dict[str, Any]):
        """
        Removes the key the given keyword argument dictionary.

        :param kwargs:
            Keyword argument dictionary.
        """
        kwargs.pop(self.curated_config_kwarg)


@dataclass
class HFConfigKeyDefault:
    """
    Represents a default value for a Hugging Face configuration
    key. Required to pass ``None`` as a valid default value.

    :param value:
        The default value for the key.
    """

    value: Any


class CommonCuratedToHFConverters:
    """
    Common functions to convert Curated Transformer config
    values to a compatible Hugging Face config format.
    """

    @staticmethod
    def n_attention_heads_uniform(config: TransformerConfig) -> int:
        assert (
            config.layer.attention.n_key_value_heads
            == config.layer.attention.n_query_heads
        )
        return config.layer.attention.n_key_value_heads

    @staticmethod
    def attention_dropout(config: TransformerConfig) -> float:
        return config.layer.attention.dropout_prob

    @staticmethod
    def activation(config: TransformerConfig) -> str:
        return config.layer.feedforward.activation.value

    @staticmethod
    def dtype(config: TransformerConfig) -> str:
        return str(torch.float32).split(".")[1]

    @staticmethod
    def embedding_width(config: TransformerConfig) -> int:
        return config.embedding.embedding_width

    @staticmethod
    def hidden_width(config: TransformerConfig) -> int:
        return config.layer.feedforward.hidden_width

    @staticmethod
    def intermediate_width(config: TransformerConfig) -> int:
        return config.layer.feedforward.intermediate_width

    @staticmethod
    def layer_norm_eps(config: TransformerConfig) -> float:
        return config.layer.layer_norm_eps

    @staticmethod
    def hidden_dropout(config: TransformerConfig) -> float:
        return config.layer.dropout_prob

    @staticmethod
    def n_pieces(config: TransformerConfig) -> int:
        return config.embedding.n_pieces

    @staticmethod
    def n_types(config: TransformerConfig) -> Optional[int]:
        return config.embedding.n_types

    @staticmethod
    def n_hidden_layers(config: TransformerConfig) -> int:
        return config.layer.n_hidden_layers

    @staticmethod
    def n_positions(config: TransformerConfig) -> Optional[int]:
        return config.embedding.n_positions


class CommonHFToCuratedConverters:
    """
    Common functions to convert Hugging Face config
    values to a compatible Curated config format.
    """

    @staticmethod
    def dtype(serialized_dtype_str: str) -> Optional[torch.dtype]:
        serialized_dtype = getattr(torch, serialized_dtype_str, None)
        if not isinstance(serialized_dtype, torch.dtype):
            raise ValueError(f"Invalid torch dtype `{serialized_dtype_str}`")
        return serialized_dtype


class CommonHFKeys:
    """
    Common Hugging Face config keys.
    """

    ATTENTION_PROBS_DROPOUT_PROB = HFConfigKey(
        "attention_probs_dropout_prob",
        "attention_probs_dropout_prob",
        # Workaround for Python 3.8 limitation that doesn't allow
        # passing/calling static methods as without a class bound.
        lambda c: CommonCuratedToHFConverters.attention_dropout(c),
    )
    DTYPE = HFConfigKey(
        "torch_dtype",
        ("dtype", lambda h: CommonHFToCuratedConverters.dtype(h)),
        lambda c: CommonCuratedToHFConverters.dtype(c),
    )
    EMBEDDING_SIZE = HFConfigKey(
        "embedding_size",
        "embedding_width",
        lambda c: CommonCuratedToHFConverters.embedding_width(c),
    )
    HIDDEN_ACT = HFConfigKey(
        "hidden_act",
        ("activation", Activation),
        lambda c: CommonCuratedToHFConverters.activation(c),
    )
    HIDDEN_SIZE = HFConfigKey(
        "hidden_size",
        "hidden_width",
        lambda c: CommonCuratedToHFConverters.hidden_width(c),
    )
    INTERMEDIATE_SIZE = HFConfigKey(
        "intermediate_size",
        "intermediate_width",
        lambda c: CommonCuratedToHFConverters.intermediate_width(c),
    )
    LAYER_NORM_EPS = HFConfigKey(
        "layer_norm_eps",
        "layer_norm_eps",
        lambda c: CommonCuratedToHFConverters.layer_norm_eps(c),
    )
    MAX_POSITION_EMBEDDINGS = HFConfigKey(
        "max_position_embeddings",
        "n_positions",
        lambda c: CommonCuratedToHFConverters.n_positions(c),
    )
    TYPE_VOCAB_SIZE = HFConfigKey(
        "type_vocab_size",
        "n_types",
        lambda c: CommonCuratedToHFConverters.n_types(c),
    )
    VOCAB_SIZE = HFConfigKey(
        "vocab_size",
        "n_pieces",
        lambda c: CommonCuratedToHFConverters.n_pieces(c),
    )
    NUM_ATTENTION_HEADS_UNIFORM = HFConfigKey(
        "num_attention_heads",
        "n_attention_heads",
        lambda c: CommonCuratedToHFConverters.n_attention_heads_uniform(c),
    )
    NUM_HIDDEN_LAYERS = HFConfigKey(
        "num_hidden_layers",
        "n_hidden_layers",
        lambda c: CommonCuratedToHFConverters.n_hidden_layers(c),
    )
    HIDDEN_DROPOUT_PROB = HFConfigKey(
        "hidden_dropout_prob",
        "hidden_dropout_prob",
        lambda c: CommonCuratedToHFConverters.hidden_dropout(c),
    )


def config_from_hf(
    model_name: str,
    hf_config: Mapping[str, Any],
    hf_keys: List[Tuple[HFConfigKey, Optional[HFConfigKeyDefault]]],
) -> Dict[str, Any]:
    """
    Convert Hugging Face configuration keys to keyword arguments for
    Curated Transformers configuration classes.

    :param model_name:
        Model name. Only used in exception messages.
    :param hf_config:
        Hugging Face model configuration.
    :param hf_keys:
        List of Hugging Face configuration key descriptor and their
        defaults. Keys without any defaults are treated as mandatory
        and an error is raised if they are not found in the config.
    :returns:
        Dictionary with keyword arguments.
    """

    missing_keys = tuple(
        sorted(
            set(k.name for k, default in hf_keys if default is None).difference(
                set(hf_config.keys())
            )
        )
    )
    if len(missing_keys) != 0:
        raise ValueError(
            f"Missing keys in Hugging Face {model_name} model config: {missing_keys}"
        )

    kwargs: Dict[str, Any] = {}
    for hf_key, default in hf_keys:
        hf_key.set_kwarg(
            hf_config.get(hf_key.name, default.value if default is not None else None),
            kwargs,
        )
    return kwargs


def config_to_hf(
    curated_config: TransformerConfig,
    hf_keys: List[HFConfigKey],
) -> Dict[str, Any]:
    """
    Convert a Curated Transformers configuration to a compatible
    Hugging Face configuration dictionary.

    :param curated_config:
        Curated Transformers model configuration.
    :param hf_keys:
        List of Hugging Face configuration key descriptors.
    :returns:
        Dictionary of a Hugging Face configuration
    """
    hf_config = {}
    for hf_key in hf_keys:
        value = hf_key.mapping_from_curated_config(curated_config)
        if value is not None:
            hf_config[hf_key.name] = value
    return hf_config


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
