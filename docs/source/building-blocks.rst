Building blocks
===============

Curated Transformers provides building blocks to create your own transformer
models.

Attention
---------

.. autoclass:: curated_transformers.models.attention.AttentionMask
   :members:

.. autoclass:: curated_transformers.models.attention.KeyValueCache
   :members:

.. autoclass:: curated_transformers.models.attention.ScaledDotProductAttention
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.attention.SelfAttention
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.attention.SelfAttentionWithRotaryEmbeddings
   :members:
   :show-inheritance:

Embeddings
----------

.. autoclass:: curated_transformers.models.embeddings.SinusoidalPositionalEmbedding
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.embeddings.RotaryEmbeddings
   :members:
   :show-inheritance:

Feed-forward layers
-------------------

.. autoclass:: curated_transformers.models.feedforward.PointwiseFeedForward
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
