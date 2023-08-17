API Compatibility
=================

Curated Transformers is a new library. It is almost certain that we have made
design errors that will require us to change the API in incompatible ways in the
future. Nevertheless, we will try to make such changes easy to manage for you
by:

- Following `Semantic Versioning`_. This means that you can pin Curated
  Transformers to ``curated-transformers>=1.0.0,<2.0.0`` and your code should
  not break.
- Documenting API changes between major versions in this document. Once the next
  major version of Curated Transformers is released, you can migrate to that
  version by taking these changes into account.

Python makes nearly every part of a project visible. However, we consider some
data structures and functions to be implementation details that should not be
used externally and are not covered by the semver promise. Therefore we limit
API compatibility to the following:

- Modules, data structures, and functions that are publicly documented in the
  documentation.
- Model checkpoints. You should be able to load any model checkpoint created
  with a major version of Curated Transfomers with all subsequent minor/patch
  versions of that major version.

.. _Semantic Versioning: https://semver.org/

Specific Constructions
----------------------

In this section we discuss how API compatibility is handled in some specific
constructions.

Enums
^^^^^

All enums should be considered non-exhaustive. This means that new variants may
be added in within a major version. If you write code that matches on enum
variants, you should also handle unknown variants (e.g. by passing it to a
Curated Transfomers function that can handle all enum variants).

Mandatory Arguments
^^^^^^^^^^^^^^^^^^^

Some functions require all arguments to be specified, for instance this is
required in the constructors of most building blocks. This poses an issue when
we need to add new mandatory arguments. We will adress this as follows:

- In the *current* major version, we add the new argument with a special default
  value. This default value is used to signal that the function should use the
  behavior from before the argument was added. This ensures that existing
  invocations of such a function will continue to work.
- The last minor release of the *current* major version will emit warnings for
  such arguments when calling code relies on the default value.
- In the *next* major version, we will remove the default value to make
  the argument mandatory.

For example, consider the constructor of
:py:class:`curated_transformers.layers.ScaledDotProductAttention`:

.. code-block:: python

   def __init__(
       self,
       *,
       dropout_prob: float,
       linear_biases: Optional[AttentionLinearBiases]
   ):

Suppose that we wanted to make it possible to use another temperature in the
attention's softmax than :math:`\sqrt{d_k}`. To do this, we could add a
temperature argument ``temperature``:

.. code-block:: python

   from curated_transformers.semver import Default, FutureMandatory

   def __init__(
       self,
       *,
       dropout_prob: float,
       linear_biases: Optional[AttentionLinearBiases]
       temperature: FutureMandatory[float] = Default
   ):

We use the generic :py:class:`~curated_transformers.semver.FutureMandatory` type
to indicate that this argument will be mandatory in the future. The
:py:class:`~curated_transformers.semver.Default` distinguishes the default from
possible values of ``T``. When the default value is not overridden, the
constructor will use the behavior that was the default before the argument was
added. This special value will also make it possible for us to add deprecation
warnings in the future.

Types Used for API Compatibility
--------------------------------

.. autoclass:: curated_transformers.semver.Default
   :members:

.. autoclass:: curated_transformers.semver.FutureMandatory
   :members:

Changes Between Major Versions
------------------------------

Version 1 to 2
^^^^^^^^^^^^^^

* The factory methods of :py:class:`~curated_transformers.layers.AttentionHeads`
  add a new ``qkv_split`` argument which is mandatory in future versions.
