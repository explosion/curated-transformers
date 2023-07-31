from typing import Generic, Iterable, List

from ..models.output import CacheT
from ..tokenizers.chunks import InputChunks
from ..tokenizers.tokenizer import TokenizerBase
from .config import GeneratorConfig
from .generator import Generator


class StringGenerator(Generic[CacheT]):
    """
    Generator wrapper that takes textual input and outputs generated strings.
    It wraps a generator and uses a tokenizer to split the input into pieces
    and decode the output pieces.
    """

    inner: Generator[CacheT]
    tokenizer: TokenizerBase

    def __init__(self, tokenizer: TokenizerBase, generator: Generator[CacheT]) -> None:
        """
        Construct a string generator.

        :param tokenizer:
            Tokenizer for piece processing.
        :param generator:
            Generator to wrap.
        """
        self.inner = generator
        self.tokenizer = tokenizer

    def __call__(
        self, prompts: Iterable[InputChunks], config: GeneratorConfig
    ) -> List[str]:
        """
        Alias for :meth:`.generate`.
        """
        return self.generate(prompts, config=config)

    def generate(
        self, prompts: Iterable[InputChunks], config: GeneratorConfig
    ) -> List[str]:
        """
        Generate text using the given prompts.

        :param prompts:
            Prompts to generate from.
        :param config:
            Generator configuraton.
        :returns:
            Strings generated for the prompts.
        """

        device = next(self.inner.model.parameters()).device
        pieces = self.tokenizer(prompts)
        ids = pieces.padded_tensor(pad_left=True, device=device)
        attention_mask = pieces.attention_mask(pad_left=True, device=device)

        piece_ids: List[List[int]] = [[] for _ in range(ids.size(0))]
        for seq_ids, outputs in self.inner(
            ids=ids, attention_mask=attention_mask, config=config
        ):
            for seq_id, seq_piece_ids in zip(seq_ids.tolist(), outputs.tolist()):
                piece_ids[seq_id].extend(seq_piece_ids)

        return self.tokenizer.decode(piece_ids)
