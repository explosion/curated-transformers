from typing import Any, Type, TypeVar
from cutlery import ByteBPEProcessor
import json
from pathlib import Path

from .bbpe_tokenizer import ByteBPETokenizer
from .tokenizer import PiecesWithIds, PostTokenizer


# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="RobertaTokenizer")


class RobertaPostTokenizer(PostTokenizer):
    def __init__(
        self,
        *,
        processor: ByteBPEProcessor,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
        unk_piece: str = "<unk>",
    ):
        """Construct a RoBERTa post-tokenizer.

        processor (ByteBPEProcessor): The processor to wrap.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        unk_piece (str): The piece to use to mark unknowns.
        """
        self.unk_piece = unk_piece
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece
        self.processor = processor

    def __call__(self, pieces_with_ids: PiecesWithIds) -> PiecesWithIds:
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

        ids = []
        for seq_ids in pieces_with_ids.ids:
            ids.append([bos_id] + seq_ids + [eos_id])
        pieces = []
        for seq_pieces in pieces_with_ids.pieces:
            pieces.append([self.bos_piece] + seq_pieces + [self.eos_piece])
        lens = []
        for seq_lens in pieces_with_ids.lens:
            lens.append([1] + seq_lens + [1])

        return PiecesWithIds(ids=ids, lens=lens, pieces=pieces)


class RobertaTokenizer(ByteBPETokenizer):
    def __init__(
        self,
        *,
        processor: ByteBPEProcessor,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
        unk_piece: str = "<unk>",
    ):
        """Construct a RoBERTa tokenizer from a cutlery byte-level BPE processor.

        processor (ByteBPEProcessor): The processor to wrap.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        unk_piece (str): The piece to use to mark unknowns.
        """
        super().__init__(processor=processor)

        self.processor = processor

        self.post_tokenizer = RobertaPostTokenizer(
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
            processor=processor,
        )

    @classmethod
    def from_files(
        cls: Type[Self],
        *,
        vocab_path: Path,
        merges_path: Path,
        bos_piece: str = "<s>",
        eos_piece: str = "</s>",
        unk_piece: str = "<unk>",
    ) -> Self:
        """Construct a tokenizer from vocabulary and merge files.

        vocab_path (Path): path to the vocabulary file.
        merges_path (Path): path to the merges file.
        bos_piece (str): The piece to use to mark the beginning of a sequence.
        eos_piece (str): The piece to use to mark the end of a sequence.
        unk_piece (str): The piece to use to mark unknowns.
        """
        processor = ByteBPEProcessor.load_from_files(
            vocab=vocab_path, merges=merges_path
        )
        return cls(
            processor=processor,
            bos_piece=bos_piece,
            eos_piece=eos_piece,
            unk_piece=unk_piece,
        )

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
