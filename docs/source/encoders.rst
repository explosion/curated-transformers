Encoders
========

Architectures
-------------

These modules represent the supported encoder-only architectures.

.. autoclass:: curated_transformers.models.ALBERTEncoder
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.BERTEncoder
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.CamemBERTEncoder
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.RoBERTaEncoder
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.XLMREncoder
   :members:
   :show-inheritance:


Downloading
-----------

Each encoder type provides a ``from_hf_hub`` function that will load a model
from Hugging Face Hub. If you want to load a encoder without committing to a 
specific encoder type, you can use the :class:`~.auto_model.AutoEncoder`
class. This class also provides a ``from_hf_hub`` method but will try to infer 
the correct type automatically.

.. autoclass:: curated_transformers.models.auto_model.AutoEncoder
   :members:

Configuration
-------------

ALBERT
^^^^^^

.. autoclass:: curated_transformers.models.albert.config.ALBERTLayerConfig
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.albert.config.ALBERTConfig
   :members:
   :show-inheritance:

BERT
^^^^

.. autoclass:: curated_transformers.models.bert.config.BERTConfig
   :members:

.. autoclass:: curated_transformers.models.bert.config.BERTLayerConfig
   :members:

.. autoclass:: curated_transformers.models.bert.config.BERTAttentionConfig
   :members:

.. autoclass:: curated_transformers.models.bert.config.BERTEmbeddingConfig
   :members:

CamemBERT
^^^^^^^^^

See :ref:`roberta config`.

.. _roberta config:

RoBERTa
^^^^^^^

.. autoclass:: curated_transformers.models.roberta.config.RoBERTaConfig
   :members:
   :show-inheritance:

XLM-RoBERTa
^^^^^^^^^^^

See :ref:`roberta config`.