from typing import Dict, Optional, Type, cast

from huggingface_hub.utils import EntryNotFoundError

from ..util.hf import TOKENIZER_JSON, get_file_metadata, get_hf_config_model_type
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
            tokenizer_cls = _get_tokenizer_class_from_config(
                name=name, revision=revision
            )

        if tokenizer_cls is None:
            try:
                model_type = get_hf_config_model_type(name=name, revision=revision)
            except EntryNotFoundError:
                pass
            else:
                tokenizer_cls = HF_MODEL_MAPPING.get(model_type, None)

        if tokenizer_cls is None:
            raise ValueError(
                f"Cannot infer tokenizer for repository '{name}' with revision '{revision}'"
            )
        else:
            # This cast is safe, because we only return tokenizers.
            return cast(
                TokenizerBase, tokenizer_cls.from_hf_hub(name=name, revision=revision)
            )


def _get_tokenizer_class_from_config(
    *, name: str, revision: str
) -> Optional[Type[FromHFHub]]:
    """
    Infer the tokenizer class from the tokenizer configuration.

    :param name:
        Model name.
    :param revision:
        Model revision.
    :returns:
        Inferred class.
    """

    try:
        tokenizer_config = get_tokenizer_config(name=name, revision=revision)
    except EntryNotFoundError:
        return None

    return HF_TOKENIZER_MAPPING.get(tokenizer_config.get("tokenizer_class", None), None)
