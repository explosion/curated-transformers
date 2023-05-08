from typing import Any, List, Type, TypeVar
from cutlery import ByteBPEProcessor
import json
from pathlib import Path

from .tokenizer import PiecesWithIds, Tokenizer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="ByteBPETokenizer")


class ByteBPETokenizer(Tokenizer):
    """Piece tokenizer using byte-level byte pair encoding
    (Gage, 1994, Sennrich et al., 2016)"""

    def __init__(
        self,
        *,
        processor: ByteBPEProcessor,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
        pad_piece: str = "<pad>",
        unk_piece: str = "<unk>",
    ):
        """Construct a tokenizer from a cutlery byte-level BPE processor.

        processor (ByteBPEProcessor): The processor to wrap.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        pad_piece (str): Padding piece.
        unk_piece (str): The piece to use to mark unknowns.
        """
        self.unk_piece = unk_piece
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece
        self.pad_piece = pad_piece
        self.processor = processor

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        vocab_path: Path,
        merges_path: Path,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
        pad_piece: str = "<pad>",
        unk_piece: str = "<unk>",
    ) -> Self:
        """Construct a tokenizer from vocabulary and merge files.

        vocab_path (Path): path to the vocabulary file.
        merges_path (Path): path to the merges file.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        pad_piece (str): Padding piece.
        unk_piece (str): The piece to use to mark unknowns.
        """
        processor = ByteBPEProcessor.load_from_files(
            vocab=vocab_path, merges=merges_path
        )
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            pad_piece=pad_piece,
            unk_piece=unk_piece,
        )

    def _tokenize(self, input: List[str]) -> PiecesWithIds:
        bos_id = self.processor.piece_id(self.bos_piece)
        if bos_id is None:
            raise ValueError(
                f"Byte BPE piece encoder vocabulary doesn't contain '{self.bos_piece}' piece"
            )

        eos_id = self.processor.piece_id(self.eos_piece)
        if eos_id is None:
            raise ValueError(
                f"Byte BPE piece encoder vocabulary doesn't contain '{self.eos_piece}' piece"
            )

        pad_id = self.processor.piece_id(self.pad_piece)
        if pad_id is None:
            raise ValueError(
                f"Byte BPE piece encoder vocabulary doesn't contain '{self.pad_piece}' piece"
            )

        ids = []
        pieces = []
        lens = []

        for text in input:
            text_ids = [bos_id]
            text_pieces = [self.bos_piece]
            text_lens = [1]

            for idx, token in enumerate(text.split(" ")):
                if idx != 0:
                    token = " " + token
                token_ids, token_pieces = self.processor.encode(token)
                text_ids.extend(token_ids)
                text_pieces.extend(token_pieces)
                text_lens.append(len(token_ids))

            text_ids.append(eos_id)
            text_pieces.append(self.eos_piece)
            text_lens.append(1)

            ids.append(text_ids)
            pieces.append(text_pieces)
            lens.append(text_lens)

        return PiecesWithIds(pad_id=pad_id, ids=ids, lens=lens, pieces=pieces)

    @classmethod
    def _convert_hf_tokenizer(cls: Type[Self], tokenizer: Any) -> Self:
        serialized = tokenizer.backend_tokenizer.to_str(True)  # type: ignore
        deserialized = json.loads(serialized)
        vocab_merges = deserialized["model"]
        merges = [tuple(merge.split(" ")) for merge in vocab_merges["merges"]]
        processor = ByteBPEProcessor(vocab_merges["vocab"], merges)
        bos_piece = tokenizer.bos_token  # type: ignore
        eos_piece = tokenizer.eos_token  # type: ignore
        unk_piece = tokenizer.unk_token  # type: ignore
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
        )
