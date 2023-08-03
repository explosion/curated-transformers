Building Blocks
===============

Curated Transformers provides building blocks to create your own transformer
models.

Embedding Layers
----------------

These modules implement full embedding layers.

.. autoclass:: curated_transformers.layers.EmbeddingDropouts
   :members:

.. autoclass:: curated_transformers.layers.EmbeddingLayerNorms
   :members:

.. autoclass:: curated_transformers.layers.TransformerEmbeddings
   :members:

Encoder/Decoder Layers
----------------------

These modules implement full encoder/decoder layers.

.. autoclass:: curated_transformers.layers.TransformerDropouts
   :members:

.. autoclass:: curated_transformers.layers.TransformerLayerNorms
   :members:

.. autoclass:: curated_transformers.layers.DecoderLayer
   :members:

.. autoclass:: curated_transformers.layers.EncoderLayer
   :members:

Attention
---------

These modules and their helper classes implement the Transformer attention mechanism.

.. autoclass:: curated_transformers.layers.QkvMode
   :members:

.. autoclass:: curated_transformers.layers.AttentionHeads
   :members:

.. autoclass:: curated_transformers.layers.AttentionMask
   :members:

.. autoclass:: curated_transformers.layers.KeyValueCache
   :members:

.. autoclass:: curated_transformers.layers.AttentionLinearBiases
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.ScaledDotProductAttention
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.SelfAttention
   :members:
   :show-inheritance:

Embeddings
----------

These modules implement various positional embeddings used by the Transformer.

.. autoclass:: curated_transformers.layers.SinusoidalPositionalEmbedding
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.RotaryEmbeddings
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.QueryKeyRotaryEmbeddings
   :members:
   :show-inheritance:

Feed-forward Layers
-------------------

.. autoclass:: curated_transformers.layers.PointwiseFeedForward
   :members:
   :show-inheritance:


Activations
-----------

.. autoclass:: curated_transformers.layers.Activation
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.GELUFast
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.layers.GELUNew
   :members:
   :show-inheritance:


Normalization
-------------

.. autoclass:: curated_transformers.layers.RMSNorm
   :members:
   :show-inheritance:


Model Outputs
-------------

These dataclasses encapsulate the outputs produced by the different modules.

.. autoclass:: curated_transformers.models.ModelOutput
   :members:

.. autoclass:: curated_transformers.models.ModelOutputWithCache
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.CausalLMOutputWithCache
   :members:
   :show-inheritance:

Model Configs
-------------

These dataclasses encapsulate the configurable parameters of the Transformer model.

.. autoclass:: curated_transformers.models.RotaryEmbeddingConfig
   :members:

.. autoclass:: curated_transformers.models.TransformerAttentionLayerConfig
   :members:

.. autoclass:: curated_transformers.models.TransformerEmbeddingLayerConfig
   :members:

.. autoclass:: curated_transformers.models.TransformerFeedForwardLayerConfig
   :members:

.. autoclass:: curated_transformers.models.TransformerLayerConfig
   :members:
