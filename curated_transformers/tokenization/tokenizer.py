from typing import List, Optional, Type, TypeVar, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from .._compat import has_hf_transformers, transformers

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="Tokenizer")


@dataclass
class PiecesWithIds:
    ids: List[List[int]]
    lens: List[List[int]]
    pieces: List[List[str]]

    @property
    def attention_mask(self):
        """CPU tensor with attention masks.

        The mask is equivalent to `ids.padded_tensor != padding_idx`."""
        n_seqs = len(self.ids)
        max_len = max(len(seq_ids) for seq_ids in self.ids)
        mask = torch.full((n_seqs, max_len), False)
        for idx, seq_ids in enumerate(self.ids):
            mask[idx, : len(seq_ids)] = True
        return mask

    def padded_tensor(self, *, padding_id: int):
        """Padded CPU tensor of the piece identifiers."""
        n_seqs = len(self.ids)
        max_len = max(len(seq_ids) for seq_ids in self.ids)
        padded = torch.full((n_seqs, max_len), padding_id, dtype=torch.int32)
        for idx, seq_ids in enumerate(self.ids):
            padded[idx, : len(seq_ids)] = torch.tensor(seq_ids)
        return padded


class PreTokenizer(ABC):
    @abstractmethod
    def __call__(self, input: List[str]) -> List[str]:
        ...


class PostTokenizer(ABC):
    @abstractmethod
    def __call__(self, pieces: PiecesWithIds) -> PiecesWithIds:
        ...


class Tokenizer(ABC):
    pre_tokenizer: Optional[PreTokenizer] = None
    post_tokenizer: Optional[PostTokenizer] = None

    def __call__(self, input: Union[str, List[str]]) -> PiecesWithIds:
        """Split one or more texts into pieces.

        input (Union[str, List[str]]): text (str) or texts (List[str])
            to split."""
        input = input if isinstance(input, list) else [input]
        if self.pre_tokenizer is not None:
            input = self.pre_tokenizer(input)

        pieces = self._tokenize(input)

        if self.post_tokenizer is not None:
            pieces = self.post_tokenizer(pieces)

        return pieces

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
