Parameter Sharing
=================

Curated Transformers allows models to share entire submodules and 
individual parameters in their inner layers. A model that requires 
the sharing of parameters/modules derives from the :py:class:`~curated_transformers.sharing.Shareable`
mixin and provides the necessary descriptors of the shared data.


.. autoclass:: curated_transformers.sharing.SharedDataType
   :members:

.. autoclass:: curated_transformers.sharing.SharedDataDescriptor
   :members:

.. autoclass:: curated_transformers.sharing.Shareable
   :members:
