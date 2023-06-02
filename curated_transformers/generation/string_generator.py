from typing import Generic, Iterable, Iterator, List, Tuple

from .generator import Generator
from ..models.attention import CacheT
from ..tokenization.chunks import InputChunks
from ..tokenization.tokenizer import Tokenizer


class StringGenerator(Generic[CacheT]):
    """
    Generator wrapper that takes textual input and outputs generated strings.
    """

    inner: Generator[CacheT]
    tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer, generator: Generator[CacheT]) -> None:
        """
        Wrap a generator, using a tokenizer to split the input into pieces
        and decode the output pieces into strings.

        :param tokenizer:
            Tokenizer for piece processing.
        :param generator:
            Generator to wrap.
        """
        self.inner = generator
        self.tokenizer = tokenizer

    def __call__(
        self, prompts: Iterable[InputChunks], eos_id: int
    ) -> Iterator[List[Tuple[int, str]]]:
        """
        See the :meth:`.generate` method.
        """
        return self.generate(prompts, eos_id)

    def generate(
        self, prompts: Iterable[InputChunks], eos_id: int
    ) -> Iterator[List[Tuple[int, str]]]:
        """
        Generate text using the given prompts. This function yields for
        each generation step a list of sequence identifiers and the
        corresponding generated substring.

        :param prompts:
            Prompts to generate from.
        :param eos_id:
            Piece identifier that signals the end of generation.
        :returns:
            An iterator returning for each generation the step sequence
            identifiers and the substrings that were generated
            for the sequences.
        """

        device = next(self.inner.model.parameters()).device
        pieces = self.tokenizer(prompts)
        ids = pieces.padded_tensor(padding_id=0, pad_left=True).to(device)
        attention_mask = pieces.attention_mask(pad_left=True).to(device)

        for seq_ids, outputs in self.inner(
            ids=ids, attention_mask=attention_mask, eos_id=eos_id
        ):
            decoded = self.tokenizer.decode(outputs.tolist())
            yield list(zip(seq_ids.tolist(), decoded))
