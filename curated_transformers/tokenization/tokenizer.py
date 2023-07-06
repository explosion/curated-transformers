import json
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union, cast

import torch
from huggingface_hub.utils import EntryNotFoundError
from tokenizers import Tokenizer as HFTokenizer
from torch import Tensor

from ..util.hf import (
    HF_TOKENIZER_CONFIG,
    SPECIAL_TOKENS_MAP,
    TOKENIZER_JSON,
    get_special_piece,
    get_special_tokens_map,
    get_tokenizer_config,
)
from ._hf_compat import clean_up_decoded_string_like_hf
from .chunks import (
    InputChunks,
    MergedInputChunks,
    MergedSpecialPieceChunk,
    SpecialPieceChunk,
    TextChunk,
)
from .hf_hub import FromHFHub

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="Tokenizer")


@dataclass
class PiecesWithIds:
    """
    Encoded output of tokenizers.

    :param ids:
        Piece identifiers of each input sequence.
    :param pieces:
        Piece strings of each input sequence.
    """

    ids: List[List[int]]
    pieces: List[List[str]]

    def attention_mask(self, *, pad_left: bool = False) -> Tensor:
        """
        CPU tensor with attention masks. The mask is equivalent to:
        ``ids.padded_tensor(padding_id) != padding_id``

        :param pad_left:
            By default sequences shorter than the longest sequence are
            right-padded. Use left-padding when set to ``True``.
        :returns:
            The attention mask.

            **Shape:** ``(batch_size, max_seq_len)``
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
        """
        Padded CPU tensor of the piece identifiers.

        :param pad_left:
            By default sequences shorter than the longest sequence are
            right-padded. Use left-padding when set to ``True``.
        :returns:
            The padded piece ids.

            **Shape:** ``(batch_size, max_seq_len)``
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


class TokenizerBase(ABC):
    """
    Base class for all tokenizers.
    """

    def __call__(
        self, input: Union[Iterable[InputChunks], Iterable[str]]
    ) -> PiecesWithIds:
        """
        Split one or more texts into pieces.

        :param input:
            Sequences to tokenize. If the sequences are strings, they are
            automatically converted to chunks.
        :returns:
            Pieces in each sequence.
        """
        return self.encode(input)

    @abstractmethod
    def decode(
        self, input: Iterable[Iterable[int]], *, skip_special_pieces: bool = True
    ) -> List[str]:
        """
        Reconstruct string sequences from piece identifiers.

        :param input:
            The piece identifiers to reconstruct the strings from.
        :param skip_special_pieces:
            Skip special pieces during decoding.
        :returns:
            The decoded strings.
        """
        ...

    @abstractmethod
    def encode(
        self, input: Union[Iterable[InputChunks], Iterable[str]]
    ) -> PiecesWithIds:
        """
        Split one or more texts into pieces.

        :param input:
            Sequences to tokenize. If the sequences are strings, they are
            automatically converted to chunks.
        :returns:
            Pieces in each sequence.
        """
        ...

    @abstractmethod
    def piece_to_id(self, piece: str) -> Optional[int]:
        """
        Get the ID for a single piece.

        :param piece:
            The piece to look up the ID for.
        :returns:
            The piece identifier or ``None`` when the piece is unknown.
        """
        ...

    @property
    @abstractmethod
    def eos_piece(self) -> Optional[str]:
        """
        Get the end-of-sequence piece.

        :returns:
            The end-of-sequence piece or ``None`` when this piece is not
            defined.
        """
        ...


class Tokenizer(TokenizerBase, FromHFHub):
    """
    This class wraps the tokenizers from the `tokenizers` package. It supports a
    wide range of piece tokenizers, including word piece, byte pair encoding, and
    sentencepiece unigram tokenizers. This is the tokenizer that should be used
    in the majority of cases. The other tokenizers in the `curated-transformers`
    package should only be used when you have a legacy tokenizer that is not in
    Hugging Face `tokenizer.json` format.
    """

    def __init__(
        self,
        *,
        tokenizer: HFTokenizer,
        config: Optional[Dict[str, Any]],
        special_tokens_map: Optional[Dict[str, Any]],
    ):
        """
        Construct a tokenizer.

        :param tokenizer:
            The ``tokenizers`` tokenizer to use.
        :param config:
            Additional tokenizer configuration.
        :param special_tokens_map:
            Map of special tokens.
        """
        self.tokenizer = tokenizer
        self._eos_piece = self._get_special_piece(
            piece_name="eos_token",
            tokenizer_config=config,
            special_tokens_map=special_tokens_map,
        )

    def _get_special_piece(
        self,
        *,
        piece_name: str,
        tokenizer_config: Optional[Dict[str, Any]],
        special_tokens_map: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Attempt to get the special piece identifier from the special tokens
        mapping or the tokenizer configuration.

        :param piece_name:
            The name of the special piece to look up.
        :param tokenizer_config:
            The tokenizer configuration.
        :param special_piece_map:
            Special pieces mapping.
        :returns:
            The special piece, if defined in the special tokens mapping or
            the tokenizer configuration.
        """
        piece = None
        if special_tokens_map is not None:
            piece = get_special_piece(special_tokens_map, piece_name)
        if piece is None and tokenizer_config is not None:
            get_special_piece(tokenizer_config, piece_name)
        return piece

    def decode(
        self,
        input: Iterable[Iterable[int]],
        *,
        skip_special_pieces: bool = True,
    ) -> List[str]:
        decoded = self.tokenizer.decode_batch(
            input, skip_special_tokens=skip_special_pieces
        )

        decoded = [clean_up_decoded_string_like_hf(string) for string in decoded]

        return decoded

    def encode(
        self, input: Union[Iterable[InputChunks], Iterable[str]]
    ) -> PiecesWithIds:
        input = cast(Union[List[InputChunks], List[str]], list(input))
        if isinstance(input[0], str):
            input = cast(List[str], input)
            return self._encode_strings(input)
        else:
            input = cast(List[InputChunks], input)
            return self._encode_chunks(input)

    def _encode_strings(self, input: Iterable[str]) -> PiecesWithIds:
        non_str = {type(seq).__name__ for seq in input if not isinstance(seq, str)}
        if non_str:
            raise ValueError(f"Non-string inputs: {', '.join(list(sorted(non_str)))}")
        encodings = self.tokenizer.encode_batch(input)
        ids = [encoding.ids for encoding in encodings]
        pieces = [encoding.tokens for encoding in encodings]
        return PiecesWithIds(ids=ids, pieces=pieces)

    def _encode_chunks(self, input: Iterable[InputChunks]) -> PiecesWithIds:
        non_str = {
            type(seq).__name__ for seq in input if not isinstance(seq, InputChunks)
        }
        if non_str:
            raise ValueError(f"Non-chunk inputs: {', '.join(list(sorted(non_str)))}")

        merged_chunks = [seq_chunks.merge_text_chunks() for seq_chunks in input]

        ids = []
        pieces = []

        for seq in merged_chunks:
            seq_ids = []
            seq_pieces = []

            for chunk in seq:
                if isinstance(chunk, MergedSpecialPieceChunk):
                    piece_id = self.tokenizer.token_to_id(chunk.piece)
                    seq_ids.append(piece_id)
                    seq_pieces.append(chunk.piece)
                else:
                    encoding = self.tokenizer.encode(chunk.text)
                    seq_ids.extend(encoding.ids)
                    seq_pieces.extend(encoding.tokens)

            ids.append(seq_ids)
            pieces.append(seq_pieces)

        return PiecesWithIds(ids=ids, pieces=pieces)

    @property
    def eos_piece(self) -> Optional[str]:
        return self._eos_piece

    @classmethod
    def from_dir(cls: Type[Self], path: Path) -> Self:
        """
        Load the tokenizer from a directory with a ``tokenizer.json`` file.

        :param path:
            Path to the tokenizer file.
        """
        tokenizer_path = path / TOKENIZER_JSON
        config_path = path / HF_TOKENIZER_CONFIG
        special_tokens_map_path = path / SPECIAL_TOKENS_MAP
        hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        config = None
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        special_tokens_map = None
        if special_tokens_map_path.is_file():
            with open(special_tokens_map_path, encoding="utf-8") as f:
                special_tokens_map = json.load(f)
        return cls(
            tokenizer=hf_tokenizer,
            config=config,
            special_tokens_map=special_tokens_map,
        )

    @classmethod
    def from_hf_hub(cls: Type[Self], *, name: str, revision: str = "main") -> Self:
        hf_tokenizer = HFTokenizer.from_pretrained(name, revision)

        try:
            config = get_tokenizer_config(name=name, revision=revision)
        except EntryNotFoundError:
            config = None
        try:
            special_tokens_map = get_special_tokens_map(name=name, revision=revision)
        except EntryNotFoundError:
            special_tokens_map = None
        return cls(
            tokenizer=hf_tokenizer,
            config=config,
            special_tokens_map=special_tokens_map,
        )

    @classmethod
    def from_json(
        cls: Type[Self],
        tokenizer_json: str,
        config_json: Optional[str] = None,
        special_tokens_map_json: Optional[str] = None,
    ) -> Self:
        """
        Load the tokenizer from a serialized JSON string..

        :param json:
            The JSON string.
        """
        hf_tokenizer = HFTokenizer.from_str(tokenizer_json)
        config = json.loads(config_json) if config_json is not None else None
        special_tokens_map = (
            json.loads(special_tokens_map_json)
            if special_tokens_map_json is not None
            else None
        )
        return cls(
            tokenizer=hf_tokenizer,
            config=config,
            special_tokens_map=special_tokens_map,
        )

    def piece_to_id(self, piece: str) -> Optional[int]:
        return self.tokenizer.token_to_id(piece)


class PreDecoder(ABC):
    """
    Callable applied before decoding.
    """

    @abstractmethod
    def __call__(self, input: Iterable[Iterable[int]]) -> List[List[int]]:
        """
        Apply the pre-decoder on the input.

        :param input:
            Piece identifiers of each input sequence.
        :returns:
            Modified piece identifiers.
        """
        ...


class PostDecoder(ABC):
    """
    Callable applied after decoding.
    """

    @abstractmethod
    def __call__(self, output: Iterable[str]) -> List[str]:
        """
        Apply the post-decoder on the output.

        :param output:
            Decoded strings from the tokenizer.
        :returns:
            Modified decoded strings.
        """
        ...


class PreEncoder(ABC):
    """
    Callable applied before encoding.
    """

    @abstractmethod
    def __call__(self, chunks: Iterable[InputChunks]) -> List[InputChunks]:
        """
        Apply the pre-encoder on the chunks.

        :param chunks:
            Input chunks of each input sequence.
        :returns:
            Modified input chunks.
        """
        ...


class PostEncoder(ABC):
    """
    Callable applied after encoding.
    """

    @abstractmethod
    def __call__(self, pieces: PiecesWithIds) -> PiecesWithIds:
        """
        Apply the post-encoder on the pieces.

        :param pieces:
            Encoded output of the tokenzier.
        :returns:
            Modified encoded output.
        """
        ...


class Normalizer(ABC):
    """
    Callable applied before encoding.
    """

    @abstractmethod
    def __call__(self, chunks: Iterable[InputChunks]) -> List[InputChunks]:
        """
        Apply the normalizer on the chunks.

        :param chunks:
            Input chunks of each input sequence.
        :returns:
            Modified input chunks.
        """
        ...


class LegacyTokenizer(TokenizerBase):
    """
    Base class for legacy tokenizers.
    """

    normalizer: Optional[Normalizer] = None
    pre_decoder: Optional[PreDecoder] = None
    post_decoder: Optional[PostDecoder] = None
    pre_encoder: Optional[PreEncoder] = None
    post_encoder: Optional[PostEncoder] = None

    def __call__(
        self, input: Union[Iterable[InputChunks], Iterable[str]]
    ) -> PiecesWithIds:
        return self.encode(input)

    def decode(
        self, input: Iterable[Iterable[int]], skip_special_pieces: bool = True
    ) -> List[str]:
        if self.pre_decoder is not None:
            input = self.pre_decoder(input)

        strings = self._decode(input, skip_special_pieces)

        if self.post_decoder is not None:
            strings = self.post_decoder(strings)

        return strings

    def encode(
        self, input: Union[Iterable[InputChunks], Iterable[str]]
    ) -> PiecesWithIds:
        chunks = self._convert_strings(input)

        if self.normalizer is not None:
            chunks = self.normalizer(chunks)
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

    @abstractmethod
    def _decode(
        self, input: Iterable[Iterable[int]], skip_special_pieces: bool
    ) -> List[str]:
        ...

    @abstractmethod
    def _encode(self, input: Iterable[MergedInputChunks]) -> PiecesWithIds:
        ...


class AddBosEosPreEncoder(PreEncoder):
    """
    Adds beginning/end of sequence markers before the encoding process.
    """

    def __init__(
        self,
        *,
        bos_piece: Optional[str],
        eos_piece: Optional[str],
    ):
        """
        Construct a pre-encoder that adds beginning/end of sequence markers.

        :param bos_piece:
            The piece used to mark the beginning of a sequence. The piece
            is not added when its value is ``None``.
        :param eos_piece:
            The piece used to mark the end of a sequence. The piece is not
            added when its value is ``None``.
        """
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece

    def __call__(self, chunks: Iterable[InputChunks]) -> List[InputChunks]:
        bos_eos_chunks = []
        for seq_chunks in chunks:
            bos_chunks = (
                [SpecialPieceChunk(self.bos_piece)]
                if self.bos_piece is not None
                else []
            )
            eos_chunks = (
                [SpecialPieceChunk(self.eos_piece)]
                if self.eos_piece is not None
                else []
            )
            bos_eos_chunks.append(InputChunks([*bos_chunks, *seq_chunks, *eos_chunks]))
        return bos_eos_chunks


class UnicodeNormalization(str, Enum):
    """
    Unicode normalization schemes.
    """

    NFC = "NFC"
    NFKC = "NFKC"
    NFD = "NFD"
    NFKD = "NFKD"


class DefaultNormalizer(Normalizer):
    """
    Performs normalization operations on input chunks before encoding.
    """

    def __init__(
        self,
        *,
        utf_normalization: Optional[UnicodeNormalization] = None,
        lowercase: bool = False,
        strip_accents: bool = False,
    ) -> None:
        """
        Construct a default normalizer.

        :param utf_normalization:
            Unicode normalization scheme to use.
        :param lowercase:
            Lowercase text.
        :param strip_accents:
            Remove accents from text.
        """
        super().__init__()
        self.utf_normalization = utf_normalization
        self.lowercase = lowercase
        self.strip_accents = strip_accents

    def __call__(self, chunks: Iterable[InputChunks]) -> List[InputChunks]:
        chunks = list(chunks)
        for chunk in chunks:
            for piece_or_text in chunk:
                if not isinstance(piece_or_text, TextChunk):
                    continue
                text = piece_or_text.text
                if self.lowercase:
                    text = text.lower()
                if self.strip_accents:
                    # Normalize with NFD to decompose accents.
                    text = unicodedata.normalize(UnicodeNormalization.NFD, text)
                    text = "".join(
                        [char for char in text if unicodedata.category(char) != "Mn"]
                    )
                if self.utf_normalization is not None:
                    # This has to be done at the end to ensure that previously
                    # applied normalization is correctly overridden.
                    text = unicodedata.normalize(self.utf_normalization, text)
                piece_or_text.text = text
        return chunks
