Development
===========

TorchScript
-----------

Tracing
^^^^^^^

We support TorchScript tracing and tracing is tested for all models when using
the ``--slow`` flag.

Tracing only accepts a small number of types for the argument and return values
of a traced module. For our purposes, these types are: ``Tensor``, ``Dict[str,
Tensor]``, or tuples of these types. This has ramifications for our models
because they take different argument types (eg. ``AttentionMask`` and
``KeyValueCache``) and return ``ModelOutput`` or one of its subclasses. What
complicates this is that we want to keep strong typing outside TorchScript. We
have addressed these issues as follows.

Our argument types are dataclasses with only ``Tensor`` fields. These types can
be represented as ``Dict[str, Tensor]`` without a loss of information. To this
end, we have made a ``DataclassAsDict`` base class. Dataclasses that inherit
from this class are also proper dictionaries. This allows us to pass these data
structures to traced models. When such a type is passed to a traced model, the
original type information is erased and inside the model the argument will be a
regular dictionary. To handle these arguments uniformly and retain access to
utility methods/properties, we rewrap the dictionary as a class. For instance, a
method that uses ``AttentionMask`` can rewrap ``Union[AttentionMask, Dict[str,
Tensor]]`` as an ``AttentionMask``:

.. code-block:: python

   attention_mask = AttentionMask.jit_rewrap(attention_mask)

The ``ModelOutput``-based return types can contain nested dataclasses. Eg.
``ModelOutputWithCache`` contains an ``Optional[List[CacheT]]`` field where
``CacheT`` can be ``KeyValueCache``. Consequently, not every ``ModelOutput`` can
be represented as a ``Dict[str, Tentor]``. For that reason we represent them as
tuples instead. We provide a ``DataclassAsTuple`` base class that makes a
dataclass convertable tuples. Since we don't want to return tuples for untraced
models, each model should only return a tuple conditional on the model being
traced:

.. code-block:: python

   if torch.jit.is_tracing():
       return output.astuple()  # type: ignore[return-value]
   else:
       return output

The type ignore is intentional, since we don't want the tuple to appear in the
return type.

Scripting
^^^^^^^^^

We do *not* support TorchScript scripting, since it would require too many
compromises to code quality (eg. we cannot use `torch.finfo`).`