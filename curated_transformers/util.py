from typing import Any, Callable, Any, List, Set, Iterable, Dict, TYPE_CHECKING
from functools import partial
from spacy.language import Language
import itertools
import thinc

if TYPE_CHECKING:
    from .pipe import Transformer

thinc.registry.create("model_loaders", entry_points=True)
registry = thinc.registry


def batch_by_length(seqs, max_words: int) -> List[List[int]]:
    """Given a list of sequences, return a batched list of indices into the
    list, where the batches are grouped by length, in descending order.
    Batches may be at most max_words in size, defined as max sequence length * size.
    """
    # Use negative index so we can get sort by position ascending.
    lengths_indices = [(len(seq), i) for i, seq in enumerate(seqs)]
    lengths_indices.sort()
    batches: List[List[int]] = []
    batch: List[int] = []
    for length, i in lengths_indices:
        if not batch:
            batch.append(i)
        elif length * (len(batch) + 1) <= max_words:
            batch.append(i)
        else:
            batches.append(batch)
            batch = [i]
    if batch:
        batches.append(batch)
    # Check lengths match
    assert sum(len(b) for b in batches) == len(seqs)
    # Check no duplicates
    seen: Set[int] = set()
    for b in batches:
        seen.update(id(item) for item in b)
    assert len(seen) == len(seqs)
    batches = [list(sorted(batch)) for batch in batches]
    batches.reverse()
    return batches


def all_equal(iterable: Iterable[Any]) -> bool:
    """Return True if all the elements are equal to each other
    (or if the input is an empty sequence), False otherwise."""
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)  # type: ignore


def gradual_transformer_unfreezing_per_pipe(
    nlp: Language,
    callback_args: Dict[str, Any],
    freeze_params: Dict[str, int],
):
    current_step = callback_args["step"]

    # Scoped import to avoid import cycles.
    from .pipe import Transformer

    for name, pipe in nlp.components:
        unfreeze_step = freeze_params.get(name)
        if unfreeze_step is None:
            continue
        elif not isinstance(pipe, Transformer):
            raise TypeError(
                "Gradual unfreezing cannot be performed on non-Transformer component '{name}'"
            )

        pipe.frozen = current_step < unfreeze_step


def gradual_transformer_unfreezing_all_pipes(
    nlp: Language, callback_args: Dict[str, Any], unfreeze_step: int
):
    current_step = callback_args["step"]

    # Scoped import to avoid import cycles.
    from .pipe import Transformer

    for _, pipe in nlp.components:
        if not isinstance(pipe, Transformer):
            continue

        pipe.frozen = current_step < unfreeze_step


def create_gradual_transformer_unfreezing(
    target_pipes: Dict[str, int]
) -> Callable[[Language, Dict[str, Any]], None]:
    """Construct a callback that can be used to gradually unfreeze the
    weights of one or more Transformer components during training. This
    can be used to prevent catastrophic forgetting during fine-tuning.

    target_pipes (Dict[str, int]):
        A dictionary whose keys and values correspond to the names of Transformer
        components and the training step at which they should be unfrozen respectively.
    """
    unfreeze_step_all_pipes = target_pipes.get("*")
    if unfreeze_step_all_pipes is not None and len(target_pipes) > 1:
        raise ValueError(
            "Wildcard operator cannot be used in conjunction with individual pipe names"
        )

    if unfreeze_step_all_pipes is not None:
        return partial(
            gradual_transformer_unfreezing_all_pipes,
            unfreeze_step=unfreeze_step_all_pipes,
        )
    else:
        return partial(
            gradual_transformer_unfreezing_per_pipe,
            freeze_params=target_pipes,
            last_unfreeze_step=max(target_pipes.values()),
        )
