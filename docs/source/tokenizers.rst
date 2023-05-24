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

.. autoclass:: curated_transformers.tokenization.InputChunks
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.SpecialPieceChunk
   :members:

.. autoclass:: curated_transformers.tokenization.TextChunk
   :members:

Pieces
------

All tokenizers decode raw strings into pieces. The pieces are
stored in a special container :class:`.PiecesWithIds`.

.. autoclass:: curated_transformers.tokenization.PiecesWithIds
   :members:
   :show-inheritance:

Tokenizers
----------

.. autoclass:: curated_transformers.tokenization.Tokenizer
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.ByteBPETokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.WordPieceTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.SentencePieceTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.GPTNeoXTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.RobertaTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.BertTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.CamembertTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.XlmrTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:
