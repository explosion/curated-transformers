Tokenization
============

Pieces
------

All tokenizers decode raw strings into pieces. The pieces are
stored in a special container :class:`.PiecesWithIds`.

.. autoclass:: curated_transformers.tokenization.PiecesWithIds
   :members:
   :show-inheritance:

Tokenizers
----------

.. autoclass:: curated_transformers.tokenization.ByteBPETokenizer
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.GPTNeoXTokenizer
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.tokenization.RobertaTokenizer
   :members:
   :show-inheritance:
