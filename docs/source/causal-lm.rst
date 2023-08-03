Causal Language Models
======================

Base Classes
------------

.. autoclass:: curated_transformers.models.module.CausalLMModule
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.models.transformer.TransformerCausalLM
   :members:
   :show-inheritance:

Architectures
-------------

These modules represent the supported causal LM architectures. Generally, every
decoder-only architecture has a corresponding causal LM architecture.

.. autoclass:: curated_transformers.models.GPTNeoXCausalLM
   :members:
   :show-inheritance:
   :inherited-members: Module

.. autoclass:: curated_transformers.models.LlamaCausalLM
   :members:
   :show-inheritance:
   :inherited-members: Module

.. autoclass:: curated_transformers.models.FalconCausalLM
   :members:
   :show-inheritance:
   :inherited-members: Module

Downloading
-----------

Each causal LM type provides a ``from_hf_hub`` function that will load a model
from Hugging Face Hub. If you want to load a causal LM without committing to a 
specific causal LM type, you can use the :class:`~.auto_model.AutoCausalLM`
class. This class also provides a ``from_hf_hub`` method but will try to infer 
the correct type automatically.

.. autoclass:: curated_transformers.models.auto_model.AutoCausalLM
   :members:


Caching
-------

Causal language models apply causal attention, meaning that the attention
mechanism only attends to preceding pieces. So, when the model predicts the next
piece, the attention and hidden representations of the pieces before it do not
change. This means we can avoid recomputing hidden representations of
already-seen pieces by caching them. This allows us to generate text in
:math:`\mathcal{O}(n^2)` time rather than :math:`\mathcal{O}(n^3)`.

Caching works by calling the causal language model with the ``store_cache``
argument.  The model will then return the cached representations as part of its
output. The cached representations can then be passed in the next call to the
language model with the ``cache`` argument::

  cache = None
  while not_done:
      ...
      output = lm(..., cache=cache, store_cache=True)
      cache = output.cache
      ...


Configuration
-------------

See :ref:`decoder model configuration<decoder config>`.
