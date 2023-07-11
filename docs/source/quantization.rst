Quantization
============

.. autoclass:: curated_transformers.quantization.quantizable.Quantizable
   :members:

bitsandbytes
------------

These classes can be used to specify the configuration for quantizing model 
parameters using the ``bitsandbytes`` library.

.. autoclass:: curated_transformers.quantization.bnb.config.Dtype4Bit
   :members:

.. autoclass:: curated_transformers.quantization.bnb.config.BitsAndBytesConfig
   :members:
