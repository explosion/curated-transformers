from typing import Any, Iterable, List, Type, TypeVar
import unicodedata
from curated_tokenizers import WordPieceProcessor
from pathlib import Path

from .wordpiece_tokenizer import WordPieceTokenizer, clean_up_decoded_string_like_hf
from .hf_hub import FromPretrainedHFTokenizer
from .tokenizer import PiecesWithIds, PreEncoder, PostEncoder, PreDecoder, PostDecoder
from .util import remove_pieces_from_sequence, add_bos_eos_to_encoding


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="BertTokenizer")


class BertPreEncoder(PreEncoder):
    def __init__(
        self,
        *,
        lowercase: bool,
        strip_accents: bool,
    ):
        """Construct a BERT pre-encoder.

        lowercase (bool): Lowercase text.
        strip_accents (bool): Strip accents from text.
        """
        self.lowercase = lowercase
        self.strip_accents = strip_accents

    def split_token_on_punctuation(self, token: str) -> List[str]:
        """Split a token on punctuation characters. For instance,
        'AWO-Mitarbeiter' is split into ['AWO', '-', 'Mitarbeiter']"""
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
        """Checks whether `char` is a punctuation character."""
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

    def strip_token_accents(self, token: str) -> str:
        # TODO move this to the normalization phase of to the tokenizer
        token = unicodedata.normalize("NFD", token)
        return "".join([char for char in token if unicodedata.category(char) != "Mn"])

    def __call__(self, input: Iterable[str]) -> List[str]:
        preprocessed = []
        for text in input:
            words = []
            for word in text.split(" "):
                if self.lowercase:
                    word = word.lower()
                if self.strip_accents:
                    word = self.strip_token_accents(word)
                word_with_punct = self.split_token_on_punctuation(word)
                words.extend(word_with_punct)
            preprocessed.append(" ".join(words))
        return preprocessed


class BertPostEncoder(PostEncoder):
    def __init__(
        self,
        *,
        bos_piece: str,
        eos_piece: str,
        unk_piece: str,
        bos_id: int,
        eos_id: int,
        unk_id: int,
    ):
        """Construct a BERT post-encoder.

        bos_piece (str): The piece used to mark the beginning of a sequence.
        eos_piece (str): The piece used to mark the end of a sequence.
        unk_piece (int): The piece used to mark unknown tokens.
        bos_id (int): The piece id used to mark the beginning of a sequence.
        eos_id (int): The piece id used to mark the end of a sequence.
        unk_id (int): The piece id used to mark unknown tokens.
        """
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece
        self.unk_piece = unk_piece
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def __call__(self, pieces_with_ids: PiecesWithIds) -> PiecesWithIds:
        pieces_with_ids = add_bos_eos_to_encoding(
            pieces_with_ids,
            bos_piece=self.bos_piece,
            eos_piece=self.eos_piece,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )

        # Replace all unknown IDs and their corresponding pieces.
        for ids, pieces in zip(pieces_with_ids.ids, pieces_with_ids.pieces):
            for i in range(len(ids)):
                piece_id = ids[i]
                if piece_id == -1:
                    ids[i] = self.unk_id
                    pieces[i] = self.unk_piece
        return pieces_with_ids


class BertPreDecoder(PreDecoder):
    def __init__(
        self,
        *,
        bos_id: int,
        eos_id: int,
    ):
        """Construct a BERT pre-decoder.

        bos_id (int): The piece id used to mark the beginning of a sequence.
        eos_id (int): The piece id used to mark the end of a sequence.
        """
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, input: Iterable[Iterable[int]]) -> List[List[int]]:
        return [
            list(remove_pieces_from_sequence(ids, (self.bos_id, self.eos_id)))
            for ids in input
        ]


class BertPostDecoder(PostDecoder):
    def __init__(
        self,
    ):
        """Construct a BERT post-decoder."""
        pass

    def __call__(self, output: Iterable[str]) -> List[str]:
        # The transformations done in the pre-encoding stage are lossy/non-reversible,
        # we can only do a selected number of transformations to massage the output
        # to look similar to the input text.
        return [clean_up_decoded_string_like_hf(string.strip()) for string in output]


class BertTokenizer(WordPieceTokenizer, FromPretrainedHFTokenizer):
    def __init__(
        self,
        *,
        processor: WordPieceProcessor,
        bos_piece: str = "[CLS]",
        eos_piece: str = "[SEP]",
        unk_piece: str = "[UNK]",
        lowercase: bool = False,
        strip_accents: bool = False,
    ):
        """Construct a Bert tokenizer from a curated tokenizers WordPiece processor.

        processor (WordPieceProcessor): The processor to wrap.
        bos_piece (str): The piece used to mark the beginning of a sequence.
        eos_piece (str): The piece used to mark the end of a sequence.
        unk_piece (str): The piece used to mark unknown tokens.
        lowercase (bool): Lowercase text.
        strip_accents (bool): Strip accents from text.
        """
        super().__init__(processor=processor)

        self.processor = processor

        bos_id = _get_piece_id_or_fail(processor, bos_piece)
        eos_id = _get_piece_id_or_fail(processor, eos_piece)
        unk_id = _get_piece_id_or_fail(processor, unk_piece)

        self.pre_encoder = BertPreEncoder(
            lowercase=lowercase, strip_accents=strip_accents
        )
        self.post_encoder = BertPostEncoder(
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
            bos_id=bos_id,
            eos_id=eos_id,
            unk_id=unk_id,
        )
        self.pre_decoder = BertPreDecoder(bos_id=bos_id, eos_id=eos_id)
        self.post_decoder = BertPostDecoder()

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        vocab_path: Path,
        bos_piece: str = "[CLS]",
        eos_piece: str = "[SEP]",
        unk_piece: str = "[UNK]",
        lowercase: bool = False,
        strip_accents: bool = False,
    ) -> Self:
        """Construct a tokenizer from the vocabulary file.

        vocab_path (Path): Path to the vocabulary file.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        unk_piece (str): The piece used to mark unknown tokens.
        lowercase (bool): Lowercase text.
        strip_accents (bool): Strip accents from text.
        """
        processor = WordPieceProcessor.from_file(vocab_path)
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
            lowercase=lowercase,
            strip_accents=strip_accents,
        )

    @classmethod
    def convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        # TODO raise an error if the hf_tokenizer is not of the expected type?
        # We'll need to import the type from transformers, though

        # Seems like we cannot get the vocab file name for a BERT vocabulary? So,
        # instead, copy the vocabulary.
        vocab = [None] * tokenizer.vocab_size  # type: ignore
        for piece, idx in tokenizer.vocab.items():  # type: ignore
            vocab[idx] = piece

        bos_piece = tokenizer.cls_token  # type: ignore
        eos_piece = tokenizer.sep_token  # type: ignore
        unk_piece = tokenizer.unk_token  # type: ignore
        lowercase = tokenizer.do_lower_case  # type: ignore
        strip_accents = tokenizer.backend_tokenizer.normalizer.strip_accents  # type: ignore
        # Huggingface BERT also strips accents when lowercasing is enabled
        # and accent stripping is not defined.
        strip_accents = strip_accents or (strip_accents is not False and lowercase)
        processor = WordPieceProcessor(vocab)
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
            lowercase=lowercase,
            strip_accents=strip_accents,
        )


def _get_piece_id_or_fail(processor: WordPieceProcessor, piece: str):
    try:
        return processor.get_initial(piece)
    except KeyError:
        raise ValueError(
            f"BERT piece encoder vocabulary doesn't contain '{piece}' piece"
        )
