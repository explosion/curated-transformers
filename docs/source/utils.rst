Utilities
=========

Context Managers
----------------

.. autofunction:: curated_transformers.layers.enable_torch_sdp

Hugging Face
------------

Loading Models from Hugging Face Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These mixin classes are used to implement support for loading models and
tokenizers directly from Hugging Face Hub.

.. attention::
   To download models hosted in private repositories, the user will first need to
   set up their authentication token using the `Hugging Face Hub client`_.

.. _Hugging Face Hub client: https://huggingface.co/docs/huggingface_hub/quick-start#login

.. autoclass:: curated_transformers.models.FromHFHub
   :members:

.. autoclass:: curated_transformers.generation.FromHFHub
   :members:

.. autoclass:: curated_transformers.tokenizers.FromHFHub
   :members:
