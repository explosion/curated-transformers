.. _generation:

Generation
==========


Models
------

These classes provide the interface for performing text generation using causal LMs.

.. autoclass:: curated_transformers.generation.generator.Generator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.string_generator.StringGenerator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.generator_wrapper.GeneratorWrapper
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.default_generator.DefaultGenerator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.dolly_v2.DollyV2Generator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.falcon.FalconGenerator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:


Downloading
-----------

Each generator type provides a ``from_hf_hub`` function that will load a model
from Hugging Face Hub. If you want to load a generator without committing to a 
specific generator type, you can use the :class:`~.auto_generator.AutoGenerator`
class. This class also provides a ``from_hf_hub`` method but will try to infer 
the correct type automatically.

.. autoclass:: curated_transformers.generation.auto_generator.AutoGenerator
   :members:


Configuration
-------------

These classes represent the different parameters used by generators.

.. autoclass:: curated_transformers.generation.config.GeneratorConfig
   :members:

.. autoclass:: curated_transformers.generation.config.GreedyGeneratorConfig
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.config.SampleGeneratorConfig
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.stop_conditions.StopCondition
   :members:

.. autoclass:: curated_transformers.generation.stop_conditions.EndOfSequenceCondition
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.stop_conditions.MaxGeneratedPiecesCondition
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.logits.LogitsTransform
   :members:

.. autoclass:: curated_transformers.generation.logits.TopKTransform
   :members:   
   :show-inheritance:

.. autoclass:: curated_transformers.generation.logits.TopPTransform
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.logits.TemperatureTransform
   :members:   
   :show-inheritance:
