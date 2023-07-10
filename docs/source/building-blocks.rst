Building Blocks
===============

Curated Transformers provides building blocks to create your own transformer
models.

Attention
---------

These modules and their helper classes implement the Transformer attention mechanism.

.. autoclass:: curated_transformers.layers.attention.QkvMode
   :members:

.. autoclass:: curated_transformers.layers.attention.QkvHeadSharing
   :members:

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

These modules implement various positional embeddings used by the Transformer.

.. autoclass:: curated_transformers.layers.embeddings.SinusoidalPositionalEmbedding
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.embeddings.RotaryEmbeddings
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.embeddings.QueryKeyRotaryEmbeddings
   :members:
   :show-inheritance:   

Feed-forward Layers
-------------------

.. autoclass:: curated_transformers.layers.feedforward.PointwiseFeedForward
   :members:
   :show-inheritance:


Model Outputs
-------------

These dataclasses encapsulate the outputs produced by the different modules.

.. autoclass:: curated_transformers.models.output.ModelOutput
   :members:

.. autoclass:: curated_transformers.models.output.ModelOutputWithCache
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.output.CausalLMOutputWithCache
   :members:
   :show-inheritance:
