Utilities
=========

Context Managers
----------------

.. autofunction:: curated_transformers.layers.attention.enable_torch_sdp

Hugging Face
------------

Loading Models from Hugging Face Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These mixin classes are used to implement support for loading models and
tokenizers directly from Hugging Face Hub.

.. autoclass:: curated_transformers.models.hf_hub.FromHFHub
   :members:

.. autoclass:: curated_transformers.generation.hf_hub.FromHFHub
   :members:

.. autoclass:: curated_transformers.tokenizers.hf_hub.FromHFHub
   :members: