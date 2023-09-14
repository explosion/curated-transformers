from typing import Any, Dict, Optional, Type, cast

from fsspec import AbstractFileSystem
from huggingface_hub.utils import EntryNotFoundError

from ..util.fsspec import get_config_model_type as get_model_type_fsspec
from ..util.fsspec import get_tokenizer_config as get_tokenizer_config_fsspec
from ..util.hf import TOKENIZER_JSON, get_config_model_type, get_file_metadata
from .hf_hub import FromHFHub, get_tokenizer_config
from .legacy.bert_tokenizer import BERTTokenizer
from .legacy.camembert_tokenizer import CamemBERTTokenizer
from .legacy.llama_tokenizer import LlamaTokenizer
from .legacy.roberta_tokenizer import RoBERTaTokenizer
from .legacy.xlmr_tokenizer import XLMRTokenizer
from .tokenizer import Tokenizer, TokenizerBase

HF_TOKENIZER_MAPPING: Dict[str, Type[FromHFHub]] = {
    "BertTokenizer": BERTTokenizer,
    "BertTokenizerFast": BERTTokenizer,
    "CamembertTokenizer": CamemBERTTokenizer,
    "CamembertTokenizerFast": CamemBERTTokenizer,
    "LlamaTokenizer": LlamaTokenizer,
    "LlamaTokenizerFast": LlamaTokenizer,
    "RobertaTokenizer": RoBERTaTokenizer,
    "RobertaTokenizerFast": RoBERTaTokenizer,
    "XLMRobertaTokenizer": XLMRTokenizer,
    "XLMRobertaTokenizerFast": XLMRTokenizer,
}

HF_MODEL_MAPPING: Dict[str, Type[FromHFHub]] = {
    "bert": BERTTokenizer,
    "camembert": CamemBERTTokenizer,
    "llama": LlamaTokenizer,
    "roberta": RoBERTaTokenizer,
    "xlm-roberta": XLMRTokenizer,
}


class AutoTokenizer:
    """
    Tokenizer loaded from the Hugging Face Model Hub.
    """

    # NOTE: We do not inherit from FromHFHub, because its from_hf_hub method
    #       requires that the return type is Self.

    @classmethod
    def from_hf_hub_to_cache(
        cls,
        *,
        name: str,
        revision: str = "main",
    ):
        """
        Download the tokenizer's serialized model, configuration and vocab files
        from Hugging Face Hub into the local Hugging Face cache directory.
        Subsequent loading of the tokenizer will read the files from disk. If the
        files are already cached, this is a no-op.

        :param name:
            Model name.
        :param revision:
            Model revision.
        """
        tokenizer_cls = _resolve_tokenizer_class_hf_hub(name, revision)
        tokenizer_cls.from_hf_hub_to_cache(name=name, revision=revision)

    @classmethod
    def from_fsspec(
        cls,
        *,
        fs: AbstractFileSystem,
        model_path: str,
        fsspec_args: Optional[Dict[str, Any]] = None,
    ) -> TokenizerBase:
        """
        Construct a tokenizer and load its parameters from an fsspec filesystem.

        :param fs:
            Filesystem.
        :param model_path:
            The model path.
        :param fsspec_args:
            Implementation-specific keyword arguments to pass to fsspec
            filesystem operations.
        :returns:
            The tokenizer.
        """
        tokenizer_cls = _resolve_tokenizer_class_fsspec(
            fs=fs, model_path=model_path, fsspec_args=fsspec_args
        )
        # This cast is safe, because we only return tokenizers.
        return cast(
            TokenizerBase,
            tokenizer_cls.from_fsspec(
                fs=fs, model_path=model_path, fsspec_args=fsspec_args
            ),
        )

    @classmethod
    def from_hf_hub(cls, *, name: str, revision: str = "main") -> TokenizerBase:
        """
        Infer a tokenizer type and load it from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :returns:
            The tokenizer.
        """

        tokenizer_cls = _resolve_tokenizer_class_hf_hub(name, revision)
        # This cast is safe, because we only return tokenizers.
        return cast(
            TokenizerBase, tokenizer_cls.from_hf_hub(name=name, revision=revision)
        )


def _get_tokenizer_class_from_config(
    tokenizer_config: Dict[str, Any]
) -> Optional[Type[FromHFHub]]:
    """
    Infer the tokenizer class from the tokenizer configuration.

    :param tokenizer_config:
        The tokenizer configuration.
    :param revision:
        Model revision.
    :returns:
        Inferred class.
    """
    return HF_TOKENIZER_MAPPING.get(tokenizer_config.get("tokenizer_class", None), None)


def _resolve_tokenizer_class_fsspec(
    fs: AbstractFileSystem,
    model_path: str,
    fsspec_args: Optional[Dict[str, Any]] = None,
) -> Type[FromHFHub]:
    fsspec_args = {} if fsspec_args is None else fsspec_args
    tokenizer_cls: Optional[Type[FromHFHub]] = None
    if fs.exists(f"{model_path}/{TOKENIZER_JSON}", **fsspec_args):
        return Tokenizer

    if tokenizer_cls is None:
        tokenizer_config = get_tokenizer_config_fsspec(
            fs=fs, model_path=model_path, fsspec_args=fsspec_args
        )
        if tokenizer_config is not None:
            tokenizer_cls = _get_tokenizer_class_from_config(tokenizer_config)

    if tokenizer_cls is None:
        try:
            model_type = get_model_type_fsspec(
                fs=fs, model_path=model_path, fsspec_args=fsspec_args
            )
        except:
            pass
        else:
            tokenizer_cls = HF_MODEL_MAPPING.get(model_type, None)

    if tokenizer_cls is None:
        raise ValueError(f"Cannot infer tokenizer for model at path: {model_path}")

    return tokenizer_cls


def _resolve_tokenizer_class_hf_hub(name: str, revision: str) -> Type[FromHFHub]:
    tokenizer_cls: Optional[Type[FromHFHub]] = None
    try:
        # We will try to fetch metadata to avoid potentially downloading
        # the tokenizer file twice (here and Tokenizer.from_hf_hub).
        get_file_metadata(filename=TOKENIZER_JSON, name=name, revision=revision)
    except EntryNotFoundError:
        pass
    else:
        tokenizer_cls = Tokenizer

    if tokenizer_cls is None:
        try:
            tokenizer_config = get_tokenizer_config(name=name, revision=revision)
        except EntryNotFoundError:
            pass
        else:
            tokenizer_cls = _get_tokenizer_class_from_config(tokenizer_config)

    if tokenizer_cls is None:
        try:
            model_type = get_config_model_type(name=name, revision=revision)
        except EntryNotFoundError:
            pass
        else:
            tokenizer_cls = HF_MODEL_MAPPING.get(model_type, None)

    if tokenizer_cls is None:
        raise ValueError(
            f"Cannot infer tokenizer for repository '{name}' with revision '{revision}'"
        )
    return tokenizer_cls
