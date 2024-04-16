import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union, cast

import torch
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import Repository
from tokenizers import Tokenizer as HFTokenizer
from torch import Tensor

from ..layers.attention import AttentionMask
from ..repository.file import RepositoryFile
from ..repository.fsspec import FsspecRepository
from ..repository.hf_hub import HfHubRepository
from ..repository.repository import Repository, TokenizerRepository
from ._hf_compat import clean_up_decoded_string_like_hf
from .chunks import InputChunks, MergedSpecialPieceChunk
from .hf_hub import FromHF

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

    def attention_mask(
        self, *, pad_left: bool = False, device: Optional[torch.device] = None
    ) -> AttentionMask:
        """
        Generate the attention masks. The mask is equivalent to:
        ``ids.padded_tensor(padding_id) != padding_id``

        :param pad_left:
            By default sequences shorter than the longest sequence are
            right-padded. Use left-padding when set to ``True``.
        :param device:
            Device on which the attention mask is created.
        :returns:
            The attention mask.

            *Shape:* ``(batch_size, max_seq_len)``
        """
        n_seqs = len(self.ids)
        max_len = max(len(seq_ids) for seq_ids in self.ids)
        mask = torch.full((n_seqs, max_len), False, device=device)
        for idx, seq_ids in enumerate(self.ids):
            if pad_left:
                mask[idx, -len(seq_ids) :] = True
            else:
                mask[idx, : len(seq_ids)] = True
        return AttentionMask(mask)

    def padded_tensor(
        self,
        *,
        padding_id: int = 0,
        pad_left: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Generate a padded tensor of the piece identifiers.

        :param padding_id:
            Piece identifier of the padding piece. The actual identifier
            generally doesn't matter when an attention mask is used (and
            as long as it is a valid vocabulary index).
        :param pad_left:
            By default sequences shorter than the longest sequence are
            right-padded. Use left-padding when set to ``True``.
        :param device:
            Device on which the padded tensor is created.
        :returns:
            The padded piece ids.

            *Shape:* ``(batch_size, max_seq_len)``
        """
        n_seqs = len(self.ids)
        max_len = max(len(seq_ids) for seq_ids in self.ids)
        padded = torch.full(
            (n_seqs, max_len), padding_id, dtype=torch.int32, device=device
        )
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
            The piece to look up the identifier for.
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


class Tokenizer(TokenizerBase, FromHF):
    """
    Wraps the tokenizers from the ``tokenizers`` package. It supports a
    wide range of piece tokenizers, including word piece, byte pair encoding, and
    sentencepiece unigram tokenizers. This is the tokenizer that should be used
    in the majority of cases. The other tokenizers in the ``curated-transformers``
    package should only be used when you have a legacy tokenizer that is not in
    Hugging Face ``tokenizer.json`` format.
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
            piece = get_special_piece(tokenizer_config, piece_name)
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
            Path to the tokenizer directory.
        """
        return cls.from_repo(FsspecRepository(LocalFileSystem(), str(path)))

    @classmethod
    def from_hf_hub_to_cache(
        cls: Type[Self],
        *,
        name: str,
        revision: str = "main",
    ):
        repo = TokenizerRepository(HfHubRepository(name, revision=revision))
        repo.tokenizer_json()

        try:
            _ = repo.tokenizer_config()
        except:
            pass
        try:
            _ = repo.special_tokens_map()
        except:
            pass

    @classmethod
    def from_repo(cls: Type[Self], repo: Repository) -> Self:
        def initialize_hf_tokenizer(tokenizer_file: RepositoryFile) -> HFTokenizer:
            with tokenizer_file.open(mode="rb") as f:
                # We have to open the file once before checking the local
                # file's existence (since the repo files are loaded lazily).
                if tokenizer_file.path is not None:
                    return HFTokenizer.from_file(tokenizer_file.path)
                else:
                    return HFTokenizer.from_buffer(f.read())

        repo = TokenizerRepository(repo)
        tokenizer_file = repo.tokenizer_json()
        hf_tokenizer = initialize_hf_tokenizer(tokenizer_file)

        try:
            config = repo.tokenizer_config()
        except OSError:
            config = None
        try:
            special_tokens_map = repo.special_tokens_map()
        except OSError:
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
        Load the tokenizer from serialized JSON strings.

        :param tokenizer_json:
            The JSON string of the serialized tokenizer.
        :param config_json:
            The JSON string of the tokenizer config.
        :param special_tokens_map_json:
            The JSON string of the special tokens map.
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


def get_special_piece(
    special_tokens_map: Dict[str, Any], piece_name: str
) -> Optional[str]:
    """
    Get a special piece from the special tokens map or the tokenizer
    configuration.

    :param special_tokens_map:
        The special tokens map.
    :param piece_name:
        The piece to look up.
    :returns:
        The piece or ``None`` if this particular piece was not defined.
    """
    piece = special_tokens_map.get(piece_name)
    if isinstance(piece, dict):
        piece = piece.get("content")
    return piece
