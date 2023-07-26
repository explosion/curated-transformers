import torch

from curated_transformers.generation.state import GeneratorState
from curated_transformers.generation.stop_conditions import (
    EndOfSequenceCondition,
    MaxGeneratedPiecesCondition,
)
from curated_transformers.layers.attention import AttentionMask


def test_end_of_sequence_condition():
    attention_mask = AttentionMask(torch.ones((2, 2), dtype=torch.bool))
    ids = torch.arange(0, 4).reshape(2, 2)
    state = GeneratorState(attention_mask=attention_mask, cache=None, prompt_ids=ids)
    state.generated_ids = torch.arange(4, 8).reshape(2, 2)

    completed_exclude = torch.zeros((2, 1), dtype=torch.bool)
    completed_include = torch.zeros((2, 1), dtype=torch.bool)
    EndOfSequenceCondition(4).update_completed(
        state=state,
        completed_exclude=completed_exclude,
        completed_include=completed_include,
    )
    assert not completed_exclude.any()
    assert not completed_include.any()

    EndOfSequenceCondition(6).update_completed(
        state=state,
        completed_exclude=completed_exclude,
        completed_include=completed_include,
    )
    assert not completed_exclude.any()
    assert not completed_include.any()

    EndOfSequenceCondition(5).update_completed(
        state=state,
        completed_exclude=completed_exclude,
        completed_include=completed_include,
    )
    assert completed_exclude.any()
    assert not completed_include.any()

    completed_exclude = torch.zeros((2, 1), dtype=torch.bool)
    EndOfSequenceCondition(7).update_completed(
        state=state,
        completed_exclude=completed_exclude,
        completed_include=completed_include,
    )
    assert completed_exclude.any()
    assert not completed_include.any()


def test_max_generated_pieces_condition():
    attention_mask = AttentionMask(torch.ones((2, 3), dtype=torch.bool))
    ids = torch.arange(0, 6).reshape(2, 3)
    state = GeneratorState(attention_mask=attention_mask, cache=None, prompt_ids=ids)
    state.generated_ids = torch.arange(4, 8).reshape(2, 2)

    completed_exclude = torch.zeros((2, 1), dtype=torch.bool)
    completed_include = torch.zeros((2, 1), dtype=torch.bool)
    MaxGeneratedPiecesCondition(3).update_completed(
        state=state,
        completed_exclude=completed_exclude,
        completed_include=completed_include,
    )
    assert not completed_exclude.any()
    assert not completed_include.any()

    MaxGeneratedPiecesCondition(2).update_completed(
        state=state,
        completed_exclude=completed_exclude,
        completed_include=completed_include,
    )
    assert not completed_exclude.any()
    assert completed_include.all()

    completed_include = torch.zeros((2, 1), dtype=torch.bool)
    MaxGeneratedPiecesCondition(1).update_completed(
        state=state,
        completed_exclude=completed_exclude,
        completed_include=completed_include,
    )
    assert not completed_exclude.any()
    assert completed_include.all()
