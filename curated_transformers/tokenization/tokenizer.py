from typing import Iterable, List, Optional, Set, TypeVar, Union
from abc import ABC, abstractmethod
from ahocorasick import Automaton
from dataclasses import dataclass
import re
import torch
from torch import Tensor

from .chunks import (
    InputChunks,
    MergedInputChunks,
    SpecialPieceChunk,
    TextChunk,
)


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="Tokenizer")


@dataclass
class PiecesWithIds:
    ids: List[List[int]]
    pieces: List[List[str]]

    def attention_mask(self, *, pad_left: bool = False) -> Tensor:
        """
        CPU tensor with attention masks.

        The mask is equivalent to:
        ``ids.padded_tensor(padding_id) != padding_id``

        :param pad_left:
            By default sequences shorter than the longest sequence are
            right-padded. Use left-padding when set to ``True``.
        :returns:
            The attention mask. **Shape:** (batch_size, max_seq_len)
        """
        n_seqs = len(self.ids)
        max_len = max(len(seq_ids) for seq_ids in self.ids)
        mask = torch.full((n_seqs, max_len), False)
        for idx, seq_ids in enumerate(self.ids):
            if pad_left:
                mask[idx, -len(seq_ids) :] = True
            else:
                mask[idx, : len(seq_ids)] = True
        return mask

    def padded_tensor(self, *, padding_id: int, pad_left: bool = False):
        """Padded CPU tensor of the piece identifiers.

        :param pad_left:
            By default sequences shorter than the longest sequence are
            right-padded. Use left-padding when set to ``True``.
        :returns:
            The padded piece ids. **Shape:** (batch_size, max_seq_len)
        """
        n_seqs = len(self.ids)
        max_len = max(len(seq_ids) for seq_ids in self.ids)
        padded = torch.full((n_seqs, max_len), padding_id, dtype=torch.int32)
        for idx, seq_ids in enumerate(self.ids):
            if pad_left:
                padded[idx, -len(seq_ids) :] = torch.tensor(seq_ids)
            else:
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
    def __call__(self, chunks: Iterable[InputChunks]) -> List[InputChunks]:
        ...


class PostEncoder(ABC):
    """Callable applied after encoding."""

    @abstractmethod
    def __call__(self, pieces: PiecesWithIds) -> PiecesWithIds:
        ...


class Tokenizer(ABC):
    """Base class for all tokenizers."""

    pre_decoder: Optional[PreDecoder] = None
    post_decoder: Optional[PostDecoder] = None
    pre_encoder: Optional[PreEncoder] = None
    post_encoder: Optional[PostEncoder] = None
    _special_tokens_automaton: Optional[Automaton] = None

    def __call__(
        self, input: Union[Iterable[InputChunks], Iterable[str]]
    ) -> PiecesWithIds:
        """Split one or more texts into pieces.

        :param input:
            Sequences to tokenize. If the sequences are strings, they are
            automatically converted to chunks.
        :returns:
            Pieces in each sequence.
        """
        return self.encode(input)

    def decode(self, input: Iterable[Iterable[int]]) -> List[str]:
        """Reconstruct string sequences from piece identifiers.

        :param input:
            The piece identifiers to reconstruct the strings from.
        :returns:
            The decoded strings.
        """
        if self.pre_decoder is not None:
            input = self.pre_decoder(input)

        strings = self._decode(input)

        if self.post_decoder is not None:
            strings = self.post_decoder(strings)

        return strings

    def encode(
        self, input: Union[Iterable[InputChunks], Iterable[str]]
    ) -> PiecesWithIds:
        """Split one or more texts into pieces.

        :param input:
            Sequences to tokenize. If the sequences are strings, they are
            automatically converted to chunks.
        :returns:
            Pieces in each sequence.
        """
        chunks = self._convert_strings(input)
        self._validate_chunks(chunks)

        if self.pre_encoder is not None:
            chunks = self.pre_encoder(chunks)

        merged_chunks = [seq_chunks.merge_text_chunks() for seq_chunks in chunks]
        pieces = self._encode(merged_chunks)

        if self.post_encoder is not None:
            pieces = self.post_encoder(pieces)

        return pieces

    def _convert_strings(
        self, input: Union[Iterable[InputChunks], Iterable[str]]
    ) -> List[InputChunks]:
        return [
            InputChunks([TextChunk(seq)]) if isinstance(seq, str) else seq
            for seq in input
        ]

    def _validate_chunks(self, input: List[InputChunks]):
        if self._special_tokens_automaton is None:
            special_tokens = self._special_tokens()
            if len(special_tokens) == 0:
                return
            self._special_tokens_automaton = automaton = Automaton()
            for special_token in special_tokens:
                automaton.add_word(special_token, special_token)
            automaton.make_automaton()

        assert self._special_tokens_automaton is not None
        for chunks in input:
            for piece_or_text in chunks:
                if isinstance(piece_or_text, SpecialPieceChunk):
                    matches = list(
                        self._special_tokens_automaton.iter_long(piece_or_text.piece)
                    )
                    msg = f"Special piece chunk `{piece_or_text.piece}` is invalid"
                    if len(matches) != 1:
                        raise ValueError(msg)

                    end_idx, value = matches[0]
                    start_idx = end_idx - len(value) + 1
                    if start_idx != 0 or end_idx != len(piece_or_text.piece) - 1:
                        raise ValueError(msg)
                elif isinstance(piece_or_text, TextChunk):
                    matches = list(
                        self._special_tokens_automaton.iter_long(piece_or_text.text)
                    )
                    if len(matches) != 0:
                        raise ValueError(
                            f"Text chunk `{piece_or_text.text}` contains a special piece"
                        )
                else:
                    raise ValueError(
                        f"Input chunk has an unexpected type `{type(piece_or_text)}`"
                    )

    @abstractmethod
    def _decode(self, input: Iterable[Iterable[int]]) -> List[str]:
        ...

    @abstractmethod
    def _encode(self, input: Iterable[MergedInputChunks]) -> PiecesWithIds:
        ...

    @abstractmethod
    def _special_tokens(self) -> Set[str]:
        """Special tokens supported by the tokenizer"""
        ...


class AddBosEosPreEncoder(PreEncoder):
    """
    Construct a decoder that adds beginning/end of sequence markers.
    """

    def __init__(
        self,
        *,
        bos_piece: str,
        eos_piece: str,
    ):
        """Construct a decoder that adds beginning/end of sequence markers..

        :param bos_piece:
            The piece used to mark the beginning of a sequence.
        :param eos_piece:
            The piece used to mark the end of a sequence.
        """
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece

    def __call__(self, chunks: Iterable[InputChunks]) -> List[InputChunks]:
        bos_eos_chunks = []
        for seq_chunks in chunks:
            bos_eos_chunks.append(
                InputChunks(
                    [
                        SpecialPieceChunk(self.bos_piece),
                        *seq_chunks,
                        SpecialPieceChunk(self.eos_piece),
                    ]
                )
            )
        return bos_eos_chunks
