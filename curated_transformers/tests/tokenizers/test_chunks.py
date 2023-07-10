from copy import deepcopy

from curated_transformers.tokenizers.chunks import (
    InputChunks,
    MergedSpecialPieceChunk,
    SpecialPieceChunk,
    TextChunk,
)


def test_text_chunk_merge():
    chunks = InputChunks(
        [
            TextChunk(" "),
            TextChunk("Hello world!"),
            TextChunk(" "),
        ]
    )

    # Ensure that we are not doing accidental modification to the
    # object itself.
    chunks_copy = deepcopy(chunks)
    chunks.merge_text_chunks()
    assert chunks == chunks_copy

    assert chunks.merge_text_chunks() == [
        TextChunk(" Hello world! "),
    ]


def test_special_text_special_chunk_merge():
    chunks = InputChunks(
        [
            SpecialPieceChunk("<s>", after=" "),
            TextChunk("Hello world!"),
            SpecialPieceChunk("</s>", before=" "),
        ]
    )

    # Ensure that we are not doing accidental modification to the
    # object itself.
    chunks_copy = deepcopy(chunks)
    chunks.merge_text_chunks()
    assert chunks == chunks_copy

    assert chunks.merge_text_chunks() == [
        MergedSpecialPieceChunk("<s>"),
        TextChunk(" Hello world! "),
        MergedSpecialPieceChunk("</s>"),
    ]


def test_special_special_chunk_merge():
    chunks = InputChunks(
        [
            SpecialPieceChunk("<s>", after=" "),
            SpecialPieceChunk("</s>", before=" "),
        ]
    )

    # Ensure that we are not doing accidental modification to the
    # object itself.
    chunks_copy = deepcopy(chunks)
    chunks.merge_text_chunks()
    assert chunks == chunks_copy

    assert chunks.merge_text_chunks() == [
        MergedSpecialPieceChunk("<s>"),
        TextChunk("  "),
        MergedSpecialPieceChunk("</s>"),
    ]
