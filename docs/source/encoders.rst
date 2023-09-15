Encoders
========

Base Classes
------------

.. autoclass:: curated_transformers.models.EncoderModule
   :members:
   :show-inheritance:
   :inherited-members: Module

.. autoclass:: curated_transformers.models.TransformerEncoder
   :members:
   :show-inheritance:
   :inherited-members: Module

Architectures
-------------

These modules represent the supported encoder-only architectures.

.. autoclass:: curated_transformers.models.ALBERTEncoder
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.BERTEncoder
   :members:
   :show-inheritance:
   :inherited-members: Module

.. autoclass:: curated_transformers.models.CamemBERTEncoder
   :members:
   :show-inheritance:
   :inherited-members: Module

.. autoclass:: curated_transformers.models.RoBERTaEncoder
   :members:
   :show-inheritance:
   :inherited-members: Module

.. autoclass:: curated_transformers.models.XLMREncoder
   :members:
   :show-inheritance:
   :inherited-members: Module


Downloading
-----------

Each encoder type provides a ``from_hf_hub`` function that will load a model
from Hugging Face Hub. If you want to load an encoder without committing to a 
specific encoder type, you can use the :class:`~.AutoEncoder`
class. This class also provides a ``from_hf_hub`` method but will try to infer 
the correct type automatically.

.. autoclass:: curated_transformers.models.AutoEncoder
   :members:
   :inherited-members:

Configuration
-------------

ALBERT
^^^^^^

.. autoclass:: curated_transformers.models.ALBERTConfig
   :members:

BERT
^^^^

.. autoclass:: curated_transformers.models.BERTConfig
   :members:


CamemBERT
^^^^^^^^^

See :ref:`roberta config`.

.. _roberta config:

RoBERTa
^^^^^^^

.. autoclass:: curated_transformers.models.RoBERTaConfig
   :members:
   :show-inheritance:

XLM-RoBERTa
^^^^^^^^^^^

See :ref:`roberta config`.
