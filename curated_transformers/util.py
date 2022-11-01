from typing import List, Set
import thinc

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
