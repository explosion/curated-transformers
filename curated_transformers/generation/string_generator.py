from typing import Generic, Iterable, List

from curated_transformers.generation.config import GeneratorConfig

from ..models.attention import CacheT
from ..tokenization.chunks import InputChunks
from ..tokenization.tokenizer import Tokenizer
from .generator import Generator


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
        self, prompts: Iterable[InputChunks], eos_id: int, config: GeneratorConfig
    ) -> List[str]:
        """
        See the :meth:`.generate` method.
        """
        return self.generate(prompts, eos_id, config=config)

    def generate(
        self, prompts: Iterable[InputChunks], eos_id: int, config: GeneratorConfig
    ) -> List[str]:
        """
        Generate text using the given prompts. This method returns the
        generated text for each prompt.

        :param prompts:
            Prompts to generate from.
        :param eos_id:
            Piece identifier that signals the end of generation.
        :param config:
            Generator configuraton.
        :returns:
            Strings generated for the prompts.
        """

        device = next(self.inner.model.parameters()).device
        pieces = self.tokenizer(prompts)
        ids = pieces.padded_tensor(padding_id=0, pad_left=True).to(device)
        attention_mask = pieces.attention_mask(pad_left=True).to(device)

        piece_ids: List[List[int]] = [[] for _ in range(ids.size(0))]
        for seq_ids, outputs in self.inner(
            ids=ids, attention_mask=attention_mask, eos_id=eos_id, config=config
        ):
            for seq_id, seq_piece_ids in zip(seq_ids.tolist(), outputs.tolist()):
                piece_ids[seq_id].extend(seq_piece_ids)

        return self.tokenizer.decode(piece_ids)
