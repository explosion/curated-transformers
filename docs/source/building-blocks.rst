Building Blocks
===============

Curated Transformers provides building blocks to create your own transformer
models.

Embedding Layers
----------------

These modules implement full embedding layers.

.. autoclass:: curated_transformers.layers.transformer.EmbeddingsDropouts
   :members:

.. autoclass:: curated_transformers.layers.transformer.EmbeddingsLayerNorms
   :members:

.. autoclass:: curated_transformers.layers.transformer.TransformerEmbeddings
   :members:

Encoder/Decoder Layers
----------------------

These modules implement full encoder/decoder layers.

.. autoclass:: curated_transformers.layers.transformer.TransformerDropouts
   :members:

.. autoclass:: curated_transformers.layers.transformer.TransformerLayerNorms
   :members:

.. autoclass:: curated_transformers.layers.transformer.DecoderLayer
   :members:

.. autoclass:: curated_transformers.layers.transformer.EncoderLayer
   :members:

Attention
---------

These modules and their helper classes implement the Transformer attention mechanism.

.. autoclass:: curated_transformers.layers.attention.QkvMode
   :members:

.. autoclass:: curated_transformers.layers.attention.AttentionHeads
   :members:

.. autoclass:: curated_transformers.layers.attention.AttentionMask
   :members:

.. autoclass:: curated_transformers.layers.cache.KeyValueCache
   :members:

.. autoclass:: curated_transformers.layers.attention.AttentionLinearBiases
   :members:
   :show-inheritance:

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


Activations
-----------

.. autoclass:: curated_transformers.layers.Activation
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.activations.GELUFast
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.activations.GELUNew
   :members:
   :show-inheritance:


Normalization
-------------

.. autoclass:: curated_transformers.layers.normalization.RMSNorm
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

Model Configs
-------------

These dataclasses encapsulate the configurable parameters of the Transformer model.

.. autoclass:: curated_transformers.models.config.RotaryEmbeddingConfig
   :members:

.. autoclass:: curated_transformers.models.config.TransformerAttentionLayerConfig
   :members:

.. autoclass:: curated_transformers.models.config.TransformerEmbeddingLayerConfig
   :members:

.. autoclass:: curated_transformers.models.config.TransformerFeedForwardLayerConfig
   :members:

.. autoclass:: curated_transformers.models.config.TransformerLayerConfig
   :members:
