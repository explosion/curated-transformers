Building blocks
===============

Curated Transformers provides building blocks to create your own transformer
models.

Attention
---------

.. autoclass:: curated_transformers.layers.attention.AttentionMask
   :members:

.. autoclass:: curated_transformers.layers.cache.KeyValueCache
   :members:

.. autoclass:: curated_transformers.layers.attention.ScaledDotProductAttention
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.attention.SelfAttention
   :members:
   :show-inheritance:

Embeddings
----------

.. autoclass:: curated_transformers.layers.embeddings.SinusoidalPositionalEmbedding
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.embeddings.RotaryEmbeddings
   :members:
   :show-inheritance:

Feed-forward layers
-------------------

.. autoclass:: curated_transformers.layers.feedforward.PointwiseFeedForward
   :members:
   :show-inheritance:


Model outputs
-------------

.. autoclass:: curated_transformers.models.output.ModelOutput
   :members:

.. autoclass:: curated_transformers.models.output.ModelOutputWithCache
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.output.CausalLMOutputWithCache
   :members:
   :show-inheritance:
