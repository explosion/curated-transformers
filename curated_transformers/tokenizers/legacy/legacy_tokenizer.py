import unicodedata
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, List, Optional, Union

from ..chunks import InputChunks, MergedInputChunks, SpecialPieceChunk, TextChunk
from ..tokenizer import PiecesWithIds, TokenizerBase


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
    ) -> List[str]: ...

    @abstractmethod
    def _encode(self, input: Iterable[MergedInputChunks]) -> PiecesWithIds: ...


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
