from typing import Generic, Iterable, Iterator, Tuple

from .greedy import GreedyGenerator
from ..models.attention import CacheT
from ..tokenization.chunks import InputChunks
from ..tokenization.tokenizer import Tokenizer


class StringDecode(Generic[CacheT]):
    """
    Generator wrapper that decodes ids that are she output of a generator to
    strings.
    """

    inner: GreedyGenerator[CacheT]
    tokenizer: Tokenizer

    def __init__(self, decode: GreedyGenerator[CacheT], tokenizer: Tokenizer) -> None:
        self.inner = decode
        self.tokenizer = tokenizer

    def __call__(
        self, prompts: Iterable[InputChunks], eos_id: int
    ) -> Iterator[Iterator[Tuple[int, str]]]:
        return self.decode(prompts, eos_id)

    def decode(
        self, prompts: Iterable[InputChunks], eos_id: int
    ) -> Iterator[Iterator[Tuple[int, str]]]:
        device = next(self.inner.model.parameters()).device
        pieces = self.tokenizer(prompts)
        ids = pieces.padded_tensor(padding_id=0, pad_left=True).to(device)
        attention_mask = pieces.attention_mask(pad_left=True).to(device)

        for seq_ids, outputs in self.inner(
            ids=ids, attention_mask=attention_mask, eos_id=eos_id
        ):
            decoded = self.tokenizer.decode(outputs.tolist())
            yield zip(seq_ids.tolist(), decoded)
