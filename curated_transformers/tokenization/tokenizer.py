from typing import Iterable, List, Optional, TypeVar, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


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

        The mask is equivalent to `ids.padded_tensor != padding_id`."""
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


class PreDecoder(ABC):
    """Callable applied before decoding."""

    @abstractmethod
    def __call__(self, input: Iterable[Iterable[int]]) -> List[List[int]]:
        ...


class PostDecoder(ABC):
    """Callable applied after decoding."""

    @abstractmethod
    def __call__(self, output: Iterable[str]) -> List[str]:
        ...


class PreEncoder(ABC):
    """Callable applied before encoding."""

    @abstractmethod
    def __call__(self, input: Iterable[str]) -> List[str]:
        ...


class PostEncoder(ABC):
    """Callable applied after encoding."""

    @abstractmethod
    def __call__(self, pieces: PiecesWithIds) -> PiecesWithIds:
        ...


class Tokenizer(ABC):
    pre_decoder: Optional[PreDecoder] = None
    post_decoder: Optional[PostDecoder] = None
    pre_encoder: Optional[PreEncoder] = None
    post_encoder: Optional[PostEncoder] = None

    def __call__(self, input: Union[str, Iterable[str]]) -> PiecesWithIds:
        """Split one or more texts into pieces.

        input (Union[str, Iterable[str]]): text (str) or texts (Iterable[str])
            to split."""
        return self.encode(input)

    def decode(self, input: Iterable[Iterable[int]]):
        """Reconstruct a string from piece identifiers.

        input (Iterable[Iterable[int]]): the piece identifiers
            to reconstruct the string from.
        """
        if self.pre_decoder is not None:
            input = self.pre_decoder(input)

        strings = self._decode(input)

        if self.post_decoder is not None:
            strings = self.post_decoder(strings)

        return strings

    def encode(self, input: Union[str, Iterable[str]]) -> PiecesWithIds:
        """Split one or more texts into pieces.

        input (Union[str, Iterable[str]]): text (str) or texts (Iterable[str])
            to split."""
        input = [input] if isinstance(input, str) else input
        if self.pre_encoder is not None:
            input = self.pre_encoder(input)

        pieces = self._encode(input)

        if self.post_encoder is not None:
            pieces = self.post_encoder(pieces)

        return pieces

    @abstractmethod
    def _decode(self, input: Iterable[Iterable[int]]) -> List[str]:
        ...

    @abstractmethod
    def _encode(self, input: Iterable[str]) -> PiecesWithIds:
        ...
