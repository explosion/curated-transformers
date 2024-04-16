from typing import Any, Dict, Optional, Type, cast

from fsspec import AbstractFileSystem

from ..repository.fsspec import FsspecArgs, FsspecRepository
from ..repository.hf_hub import HfHubRepository
from ..repository.repository import Repository, TokenizerRepository
from .hf_hub import FromHF
from .legacy.bert_tokenizer import BERTTokenizer
from .legacy.camembert_tokenizer import CamemBERTTokenizer
from .legacy.llama_tokenizer import LlamaTokenizer
from .legacy.roberta_tokenizer import RoBERTaTokenizer
from .legacy.xlmr_tokenizer import XLMRTokenizer
from .tokenizer import Tokenizer, TokenizerBase

HF_TOKENIZER_MAPPING: Dict[str, Type[FromHF]] = {
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

HF_MODEL_MAPPING: Dict[str, Type[FromHF]] = {
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

    # NOTE: We do not inherit from FromHF, because its from_hf_hub method
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
        tokenizer_cls = _resolve_tokenizer_class(
            TokenizerRepository(HfHubRepository(name, revision=revision))
        )
        tokenizer_cls.from_hf_hub_to_cache(name=name, revision=revision)

    @classmethod
    def from_fsspec(
        cls,
        *,
        fs: AbstractFileSystem,
        model_path: str,
        fsspec_args: Optional[FsspecArgs] = None,
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
        return cls.from_repo(FsspecRepository(fs, model_path, fsspec_args))

    @classmethod
    def from_repo(cls, repo: Repository) -> TokenizerBase:
        """
        Construct and load a tokenizer from a repository.

        :param repository:
            The repository to load from.
        :returns:
            Loaded tokenizer.
        """
        tokenizer_cls = _resolve_tokenizer_class(TokenizerRepository(repo))
        return cast(
            TokenizerBase,
            tokenizer_cls.from_repo(repo),
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
        return cls.from_repo(HfHubRepository(name, revision=revision))


def _get_tokenizer_class_from_config(
    tokenizer_config: Dict[str, Any]
) -> Optional[Type[FromHF]]:
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


def _resolve_tokenizer_class(
    repo: TokenizerRepository,
) -> Type[FromHF]:
    tokenizer_file = repo.tokenizer_json()
    if tokenizer_file.exists():
        return Tokenizer

    cls: Optional[Type[FromHF]] = None
    try:
        tokenizer_config = repo.tokenizer_config()
        cls = _get_tokenizer_class_from_config(tokenizer_config)
    except:
        pass

    if cls is None:
        try:
            model_type = repo.model_type()
            print(model_type)
            cls = HF_MODEL_MAPPING.get(model_type)
        except:
            pass

    if cls is None:
        raise ValueError(f"Cannot infer tokenizer for repo: {repo.pretty_path()}")

    return cls
