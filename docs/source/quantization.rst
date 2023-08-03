Quantization
============

.. autoclass:: curated_transformers.quantization.Quantizable
   :members:

bitsandbytes
------------

These classes can be used to specify the configuration for quantizing model 
parameters using the ``bitsandbytes`` library.

.. autoclass:: curated_transformers.quantization.bnb.Dtype4Bit
   :members:

.. autoclass:: curated_transformers.quantization.bnb.BitsAndBytesConfig
   :members:
