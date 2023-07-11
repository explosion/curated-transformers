Tokenizers
==========

Inputs
------

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

Outputs
-------

All tokenizers encode raw strings into pieces. The pieces are
stored in a special container :class:`.PiecesWithIds`.

.. autoclass:: curated_transformers.tokenizers.PiecesWithIds
   :members:
   :show-inheritance:

The encoded pieces can be decoded to produce raw strings.

Downloading
-----------

Each tokenizer type provides a ``from_hf_hub`` function that will load a
tokenizer from Hugging Face Hub. If you want to load a tokenizer without
committing to a specific tokenizer type, you can use the :class:`.AutoTokenizer`
class. This class also provides a ``from_hf_hub`` method to load a tokenizer,
but will try to infer the tokenizer type automatically.

.. autoclass:: curated_transformers.tokenizers.AutoTokenizer
   :members:

Architectures
-------------

Tokenizer architectures are separated into two layers: non-legacy tokenizers and 
legacy tokenizers. Non-legacy tokenizers wrap tokenizers from the `Hugging Face tokenizers`_ 
library, whereas legacy tokenizers wrap model-specific tokenizers bundled with the
`Hugging Face transformers`_ library.

.. _Hugging Face tokenizers: https://github.com/huggingface/tokenizers
.. _Hugging Face transformers: https://github.com/huggingface/transformers


.. autoclass:: curated_transformers.tokenizers.TokenizerBase
   :members:
   :special-members: __call__
   :show-inheritance:

Non-Legacy
^^^^^^^^^^

.. autoclass:: curated_transformers.tokenizers.Tokenizer
   :members:
   :special-members: __call__
   :show-inheritance:

Legacy
^^^^^^

.. autoclass:: curated_transformers.tokenizers.legacy.legacy_tokenizer.LegacyTokenizer
   :members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.legacy.legacy_tokenizer.PreEncoder
   :members:
   :special-members: __call__

.. autoclass:: curated_transformers.tokenizers.legacy.legacy_tokenizer.PostEncoder
   :members:
   :special-members: __call__

.. autoclass:: curated_transformers.tokenizers.legacy.legacy_tokenizer.PreDecoder
   :members:
   :special-members: __call__

.. autoclass:: curated_transformers.tokenizers.legacy.legacy_tokenizer.PostDecoder
   :members:
   :special-members: __call__

.. autoclass:: curated_transformers.tokenizers.legacy.ByteBPETokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.legacy.WordPieceTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.legacy.SentencePieceTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

Model-Specific Tokenizers
"""""""""""""""""""""""""

.. autoclass:: curated_transformers.tokenizers.legacy.BERTTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.legacy.CamemBERTTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.legacy.RoBERTaTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.legacy.LLaMATokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.tokenizers.legacy.XLMRTokenizer
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:
