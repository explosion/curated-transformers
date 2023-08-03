Development
===========

Branches
--------

We use two branches during regular development. If the current version is
*1.0.3*, then the active branches are:

- ``main``: the development branch that will lead up to *1.1.0*.
- ``v1.0.x``: the bugfix branch that will lead up to *1.0.4*.

Following semver, only bug fixes must be pushed to ``v1.0.x``. When applicable,
a bug should first be fixed in the ``main`` branch using a PR. After that, a
backport PR should be made for the ``v1.0.x`` branch with the *backport* label.

TorchScript
-----------

Tracing
^^^^^^^

We support TorchScript tracing and test it with all models when using
the ``--slow`` flag.

Tracing only accepts a small number of types for the arguments and return values
of a traced module. For our purposes, these types are: ``Tensor``, ``Dict[str,
Tensor]``, or tuples of these types. This has ramifications for our models
because they take different argument types (e.g., ``AttentionMask`` and
``KeyValueCache``) and return ``ModelOutput`` or one of its subclasses. What
complicates this is that we want to keep strong typing outside TorchScript. We
have addressed these issues as described below.

Module Arguments
""""""""""""""""

Our argument types are dataclasses with only ``Tensor`` fields. These types can
be represented as ``Dict[str, Tensor]`` without any loss of information. To this
end, we have made a ``DataclassAsDict`` base class. Dataclasses that inherit
from this class are also proper dictionaries. This allows us to pass these data
structures to traced models. When such a type is passed to a traced model, the
original type information is erased and inside the model, the argument will be a
regular dictionary. To handle these arguments uniformly and retain access to
utility methods and properties, we rewrap the dictionary as a class. For instance, a
method that uses ``AttentionMask`` can rewrap ``Union[AttentionMask, Dict[str,
Tensor]]`` as an ``AttentionMask``:

.. code-block:: python

   attention_mask = AttentionMask.jit_rewrap(attention_mask)

Module Return Values
""""""""""""""""""""

The ``ModelOutput``-based return types can contain nested dataclasses. For
instance, ``ModelOutputWithCache`` contains an ``Optional[List[CacheT]]`` field
where ``CacheT`` can be ``KeyValueCache``. Consequently, not every
``ModelOutput`` can be represented as a ``Dict[str, Tentor]``. For that reason,
we represent model outputs as tuples instead. Dataclasses that inherit from
``DataclassAsTuple`` are also a tuple.

Scripting
^^^^^^^^^

We **do not** support TorchScript scripting, since it would require too many
compromises to code quality (e.g., we cannot use ``torch.finfo``).
