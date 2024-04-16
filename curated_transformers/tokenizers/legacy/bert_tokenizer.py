import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, TypeVar

from curated_tokenizers import WordPieceProcessor

from ...repository.file import RepositoryFile
from .._hf_compat import clean_up_decoded_string_like_hf, tokenize_chinese_chars_bert
from ..chunks import (
    InputChunks,
    MergedInputChunks,
    MergedSpecialPieceChunk,
    SpecialPieceChunk,
    TextChunk,
)
from ..hf_hub import LegacyFromHF
from ..tokenizer import PiecesWithIds
from ..util import remove_pieces_from_sequence
from .legacy_tokenizer import (
    DefaultNormalizer,
    Normalizer,
    PostDecoder,
    PreDecoder,
    PreEncoder,
)
from .wordpiece_tokenizer import WordPieceTokenizer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="BERTTokenizer")

TOKENIZER_CONFIG_MAPPING: Dict[str, str] = {
    "do_lower_case": "lowercase",
    "strip_accents": "strip_accents",
    "cls_token": "bos_piece",
    "sep_token": "eos_piece",
    "unk_token": "unk_piece",
}


class BERTPreEncoder(PreEncoder):
    def __init__(
        self,
        *,
        bos_piece: str,
        eos_piece: str,
    ):
        """
        Construct a BERT pre-encoder.

        :param bos_piece:
            The piece used to mark the beginning of a sequence.
        :param eos_piece:
            The piece used to mark the end of a sequence.
        """
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece

    def split_token_on_punctuation(self, token: str) -> List[str]:
        """
        Split a token on punctuation characters. For instance,
        'AWO-Mitarbeiter' is split into ['AWO', '-', 'Mitarbeiter'].
        """
        tokens = []
        in_word = False
        while token:
            char = token[0]
            token = token[1:]
            if self.is_punctuation(char):
                tokens.append([char])
                in_word = False
            else:
                if in_word:
                    tokens[-1].append(char)
                else:
                    tokens.append([char])
                    in_word = True
        return ["".join(t) for t in tokens]

    def is_punctuation(self, char: str) -> bool:
        """
        Checks whether `char` is a punctuation character.
        """
        # ASCII punctuation from HF tranformers, since we need to split
        # in the same way.
        cp = ord(char)
        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
            return True

        return unicodedata.category(char).startswith("P")

    def __call__(self, input: Iterable[InputChunks]) -> List[InputChunks]:
        preprocessed = []

        for seq in input:
            processed_seq = InputChunks([SpecialPieceChunk(self.bos_piece)])
            for chunk in seq:
                if isinstance(chunk, TextChunk):
                    words = []
                    for word in chunk.text.split(" "):
                        word_with_punct = self.split_token_on_punctuation(word)
                        words.extend(word_with_punct)
                    processed_seq.append(TextChunk(" ".join(words)))
                else:
                    processed_seq.append(chunk)
            processed_seq.append(SpecialPieceChunk(self.eos_piece))
            preprocessed.append(processed_seq)

        return preprocessed


class BERTPreDecoder(PreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
    ):
        """
        Construct a BERT pre-decoder.

        :param bos_id:
            The piece id used to mark the beginning of a sequence.
        :param eos_id:
            The piece id used to mark the end of a sequence.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, input: Iterable[Iterable[int]]) -> List[List[int]]:
        return [
            list(remove_pieces_from_sequence(ids, (self.bos_id, self.eos_id)))
            for ids in input
        ]


class BERTPostDecoder(PostDecoder):
    def __init__(
        self,
    ):
        """
        Construct a BERT post-decoder.
        """
        pass

    def __call__(self, output: Iterable[str]) -> List[str]:
        # The transformations done in the pre-encoding stage are lossy/non-reversible,
        # we can only do a selected number of transformations to massage the output
        # to look similar to the input text.
        return [clean_up_decoded_string_like_hf(string.strip()) for string in output]


class BERTNormalizer(Normalizer):
    """
    Performs BERT normalization operations on input chunks before encoding.
    """

    def __init__(
        self,
        *,
        lowercase: bool = False,
        strip_accents: bool = False,
        tokenize_chinese_chars: bool = True,
    ):
        """
        Construct a default normalizer.

        :param lowercase:
            Lowercase text.
        :param strip_accents:
            Remove accents from text.
        :param tokenize_chinese_chars:
            Tokenize Chinese characters.
        """

        self.default_normalizer = DefaultNormalizer(
            lowercase=lowercase, strip_accents=strip_accents
        )
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def __call__(self, chunks: Iterable[InputChunks]) -> List[InputChunks]:
        chunks = self.default_normalizer(chunks)
        if self.tokenize_chinese_chars:
            for chunk in chunks:
                for piece_or_text in chunk:
                    if not isinstance(piece_or_text, TextChunk):
                        continue
                    piece_or_text.text = tokenize_chinese_chars_bert(piece_or_text.text)
        return chunks


class BERTTokenizer(WordPieceTokenizer, LegacyFromHF):
    """
    Legacy tokenizer for BERT (`Devlin et al., 2018`_) models.

    .. _Devlin et al., 2018 : https://arxiv.org/abs/1810.04805
    """

    vocab_files: Dict[str, str] = {"vocab": "vocab.txt"}

    def __init__(
        self,
        *,
        vocab: Dict[str, int],
        special_pieces: Optional[Dict[str, int]] = None,
        bos_piece: str = "[CLS]",
        eos_piece: str = "[SEP]",
        unk_piece: str = "[UNK]",
        lowercase: bool = False,
        strip_accents: bool = False,
        tokenize_chinese_chars: bool = True,
    ):
        """
        Construct a BERT tokenizer from a ``curated-tokenizers`` WordPiece processor.

        :param vocab:
            The word piece vocabulary.
        :param special_pieces:
            Special pieces.
        :param bos_piece:
            The piece used to mark the beginning of a sequence.
        :param eos_piece:
            The piece used to mark the end of a sequence.
        :param unk_piece:
            The piece used to mark unknown tokens.
        :param lowercase:
            Lowercase text.
        :param strip_accents:
            Strip accents from text.
        :param tokenize_chinese_chars:
            Tokenize Chinese characters.
        """
        super().__init__(vocab=vocab, special_pieces=special_pieces)

        self.normalizer = BERTNormalizer(
            lowercase=lowercase,
            strip_accents=strip_accents,
            tokenize_chinese_chars=tokenize_chinese_chars,
        )

        bos_id = _get_piece_id_or_fail(self.processor, bos_piece)
        eos_id = _get_piece_id_or_fail(self.processor, eos_piece)
        unk_id = _get_piece_id_or_fail(self.processor, unk_piece)

        self.unk_id = unk_id
        self.unk_piece = unk_piece
        self._eos_piece = eos_piece

        self.pre_encoder = BERTPreEncoder(
            bos_piece=bos_piece,
            eos_piece=eos_piece,
        )
        self.pre_decoder = BERTPreDecoder(bos_id=bos_id, eos_id=eos_id)
        self.post_decoder = BERTPostDecoder()

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        vocab_file: RepositoryFile,
        bos_piece: str = "[CLS]",
        eos_piece: str = "[SEP]",
        unk_piece: str = "[UNK]",
        lowercase: bool = False,
        strip_accents: bool = False,
    ) -> Self:
        """
        Construct a tokenizer from the vocabulary file.

        :param vocab_file:
            The vocabulary file.
        :param bos_piece:
            The piece to use to mark the beginning of a sequence.
        :param eos_piece:
            The piece to use to mark the end of a sequence.
        :param unk_piece:
            The piece used to mark unknown tokens.
        :param lowercase:
            Lowercase text.
        :param strip_accents:
            Strip accents from text.
        """
        vocab: Dict[str, int] = {}
        with vocab_file.open(mode="r", encoding="utf8") as f:
            for line in f:
                vocab[line.strip()] = len(vocab)

        # Add the standard special pieces to the special pieces dict.
        special_pieces = {
            piece: vocab[piece] for piece in [bos_piece, eos_piece, unk_piece]
        }

        return cls(
            vocab=vocab,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
            lowercase=lowercase,
            strip_accents=strip_accents,
            special_pieces=special_pieces,
        )

    @property
    def eos_piece(self) -> Optional[str]:
        return self._eos_piece

    @classmethod
    def _load_from_vocab_files(
        cls: Type[Self],
        *,
        vocab_files: Mapping[str, RepositoryFile],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> Self:
        extra_kwargs = {}
        if tokenizer_config is not None:
            for hf_key, curated_key in TOKENIZER_CONFIG_MAPPING.items():
                value = tokenizer_config.get(hf_key)
                if value is not None:
                    extra_kwargs[curated_key] = value
            if "strip_accents" in extra_kwargs:
                strip_accents = extra_kwargs["strip_accents"]
                lowercase = extra_kwargs.get("lowercase", False)
                # Huggingface BERT also strips accents when lowercasing is enabled
                # and accent stripping is not defined.
                extra_kwargs["strip_accents"] = strip_accents or (
                    strip_accents is not False and lowercase
                )

        return cls.from_files(vocab_file=vocab_files["vocab"], **extra_kwargs)

    def _encode(self, input: Iterable[MergedInputChunks]) -> PiecesWithIds:
        ids = []
        pieces = []

        for seq in input:
            seq_ids = []
            seq_pieces = []

            for chunk in seq:
                if isinstance(chunk, MergedSpecialPieceChunk):
                    seq_ids.append(self.processor.get_initial(chunk.piece))
                    seq_pieces.append(chunk.piece)
                else:
                    for token in chunk.text.split(" "):
                        token_ids, token_pieces = self.processor.encode(token)
                        # Replace tokens with an unknown piece by a single
                        # unknown piece. Ideally, we'd use the pieces that
                        # we found, but this is how existing BERT models
                        # treat unknowns, so we should probably do the same.
                        if any(token_id == -1 for token_id in token_ids):
                            token_ids = [self.unk_id]
                            token_pieces = [self.unk_piece]
                        seq_ids.extend(token_ids)
                        seq_pieces.extend(token_pieces)

            ids.append(seq_ids)
            pieces.append(seq_pieces)

        return PiecesWithIds(ids=ids, pieces=pieces)


def _get_piece_id_or_fail(processor: WordPieceProcessor, piece: str):
    try:
        return processor.get_initial(piece)
    except KeyError:
        raise ValueError(
            f"BERT piece encoder vocabulary doesn't contain '{piece}' piece"
        )
