Tokenization
============

Tokenizer inputs
----------------

Each tokenizer accepts a ``Iterable[str]`` or a ``Iterable[InputChunks]``. In
most cases, passing a list of strings should suffice. However, passing
:class:`.InputChunks` can be useful when special pieces need to be added
to the input.

When the tokenizer is called with a list of strings, each string is
automatically converted to a :class:`.TextChunk`, which represents a text chunk
that should be tokenized. The other type of supported chunk is the
:class:`.SpecialPieceChunk`. The piece stored by this type of chunk is not
tokenized but looked up directly.

.. autoclass:: curated_transformers.tokenizers.InputChunks
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.SpecialPieceChunk
   :members:

.. autoclass:: curated_transformers.tokenizers.TextChunk
   :members:

Pieces
------

All tokenizers decode raw strings into pieces. The pieces are
stored in a special container :class:`.PiecesWithIds`.

.. autoclass:: curated_transformers.tokenizers.PiecesWithIds
   :members:
   :show-inheritance:

Downloading tokenizers
----------------------

Each tokenizer type provides a ``from_hf_hub`` function that will load a
tokenizer from Hugging Face Hub. If you want to load a tokenizer without
committing to a specific tokenizer type, you can use the :class:`.AutoTokenizer`
class. This class also provides a ``from_hf_hub`` method to load a tokenizer,
but will try to infer the tokenizer type automatically.

.. autoclass:: curated_transformers.tokenizers.AutoTokenizer
   :members:

Tokenizers
----------

.. autoclass:: curated_transformers.tokenizers.Tokenizer
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.TokenizerBase
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.ByteBPETokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.WordPieceTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.SentencePieceTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.RobertaTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.BertTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.CamembertTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.LLaMATokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.XlmrTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:
