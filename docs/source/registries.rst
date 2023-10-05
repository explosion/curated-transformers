Registries
==========

All models in Curated Transformers are added to a registry. Each auto class uses
a registry to query which models are available. This mechanism allows
third-party models to hook into the auto classes. This makes it possible to use
construction methods such as ``AutoModel.from_hf_hub`` with third-party models.

Third-party packages can register models in the ``options.entry_points`` section
of ``setup.cfg``. For example, if the ``models`` module of the
``extra-transformers`` package contains the ``FooCausalLM``, ``BarDecoder``, and
``BazEncoder`` classes, they can be registered in ``setup.cfg`` as follows:

.. code-block:: ini

   [options.entry_points]
   curated_transformers_causal_lms =
       extra-transformers.FooCausalLM = extra_transformers.models:FooCausalLM

   curated_transformers_decoders =
       extra-transformers.BarDecoder = extra_transformers.models:BarDecoder

   curated_transformers_encoders =
       extra-transformers.BazEncoder = extra_transformers.models:BazEncoder
