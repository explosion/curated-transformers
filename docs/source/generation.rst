.. _generation:

Generation
==========


Models
------

These classes provide the interface for performing text generation using causal LMs.

.. autoclass:: curated_transformers.generation.Generator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.StringGenerator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.GeneratorWrapper
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.DefaultGenerator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.DollyV2Generator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.FalconGenerator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:

.. autoclass:: curated_transformers.generation.LlamaGenerator
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:   

.. autoclass:: curated_transformers.generation.MPTGenerator
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

.. autoclass:: curated_transformers.generation.AutoGenerator
   :members:
   :inherited-members:


Configuration
-------------

These classes represent the different parameters used by generators.

.. autoclass:: curated_transformers.generation.GeneratorConfig
   :members:

.. autoclass:: curated_transformers.generation.GreedyGeneratorConfig
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.SampleGeneratorConfig
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.StopCondition
   :members:

.. autoclass:: curated_transformers.generation.CompoundStopCondition
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.EndOfSequenceCondition
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.MaxGeneratedPiecesCondition
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.LogitsTransform
   :members:

.. autoclass:: curated_transformers.generation.CompoundLogitsTransform
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.TopKTransform
   :members:   
   :show-inheritance:

.. autoclass:: curated_transformers.generation.TopPTransform
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.generation.TemperatureTransform
   :members:   
   :show-inheritance:

.. autoclass:: curated_transformers.generation.VocabMaskTransform
   :members:   
   :show-inheritance:
