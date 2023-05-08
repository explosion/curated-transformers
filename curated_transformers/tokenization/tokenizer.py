from typing import List, Type, TypeVar, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .._compat import has_hf_transformers, transformers

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="Tokenizer")


@dataclass
class PiecesWithIds:
    ids: List[List[int]]
    lens: List[List[int]]
    pieces: List[List[str]]


class Tokenizer(ABC):
    def __call__(self, input: Union[str, List[str]]) -> PiecesWithIds:
        """Split one or more texts into pieces.

        input (Union[str, List[str]]): text (str) or texts (List[str])
            to split."""
        input = input if isinstance(input, list) else [input]
        return self._tokenize(input)

    @classmethod
    def from_hf_hub(cls: Type[Self], *, name: str, revision: str = "main") -> Self:
        """Load the tokenizer from Hugging Face hub.

        name (str): name of the tokenizer to load.
        revision: (str): revision of the tokenizer to load."""
        if not has_hf_transformers:
            raise ValueError(
                "`Loading models from Hugging Face Hub requires `transformers` package to be installed"
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        return cls._convert_hf_tokenizer(tokenizer)

    @classmethod
    @abstractmethod
    def _convert_hf_tokenizer(cls: Type[Self], tokenizer) -> Self:
        ...

    @abstractmethod
    def _tokenize(self, input: List[str]) -> PiecesWithIds:
        ...
