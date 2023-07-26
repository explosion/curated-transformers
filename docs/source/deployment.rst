Deployment
==========

In many cases, a PyTorch model can be deployed directly in a Python application,
eg. in a `Flask-based REST service
<https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html>`_. In
other cases, such as deployment to non-CUDA accelerators, additional model
transformations might be needed. On this page, we cover several deployment
scenarios.

TorchScript tracing
-------------------

Many deployment methods start from `TorchScript`_. For instance, ONNX conversion
converts the TorchScript representation of a model. TorchScript is a
statically-typed subset of Python. It only supports the types that are necessary
for representing neural network models.

Curated Transformers supports TorchScript through `tracing`_.
Tracing runs the model with some example inputs and records the computation
graph. The TorchScript code is then generated from this computation graph,
discarding all other Python code.

.. _TorchScript: https://pytorch.org/docs/stable/jit.html
.. _tracing: https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace

Tracing a model
^^^^^^^^^^^^^^^

Models are traced using the |torch.jit.trace|_ function. The first argument to
this function is the model that you would like to trace, the second argument the
inputs as a tuple. For example, we can trace a decoder as follows:


.. code-block:: python

   import torch
   import torch.jit
   from curated_transformers.models import AutoDecoder

   device = torch.device("cuda", index=0)
   decoder = AutoDecoder.from_hf_hub(name="tiiuae/falcon-7b", device=device)
   X_example = torch.zeros(4, 20, dtype=torch.long, device=device)
   traced = torch.jit.trace(decoder, (X_example),))

As you can see, we are feeding the model with an all-zeros tensor during
tracing. This is not really an issue, as long as the inputs allows the model to
run normally, tracing can do its work.

In the example above, ``traced`` is a TorchScript module. From the surface, it
behaves like any other module. We can feed it some piece identifiers to get
their hidden representations:

.. code-block:: python

   from curated_transformers.tokenizers import AutoTokenizer

   tokenizer = AutoTokenizer.from_hf_hub(name="tiiuae/falcon-7b")
   pieces = tokenizer(["Hello world!", "This is a test"])
   Y = traced(pieces.padded_tensor(padding_id=0).to(device))
   assert isinstance(Y, tuple)
   last_layer = Y[0][-1]

The model works as before, with one catch. Normally a decoder returns a
:py:class:`~curated_transformers.models.output.ModelOutputWithCache` instance,
whereas the traced model returns a tuple. The reason is that TorchScript only
supports a limited set of types. Since arbitrary types are not supported, we
convert the :py:class:`~curated_transformers.models.output.ModelOutputWithCache`
type to a tuple in a traced model. The tuple will have the same ordering as the
fields in the untraced model's output, excluding fields that are set to
``None``. In this case we don't ask the decoder to return a key-value cache, so
the ``cache`` field is ``None`` and will not be represented in the tuple.

The example above does not give correct outputs yet. Since the input sequences
have different lengths, we have to give the model an attention mask as well to
avoid that the model attends to padding pieces. This is a fairly straightforward
fix — since the attention mask is the second argument to the model, we'll
retrace the model with a dummy attention mask:

.. code-block:: python

   from curated_transformers.layers.attention import AttentionMask

   mask_example = AttentionMask(torch.ones((4, 20), dtype=torch.bool, device=device))
   traced = torch.jit.trace(decoder, (X_example, mask_example,))
   Y = traced(pieces.padded_tensor(padding_id=0).to(device),
              AttentionMask(pieces.attention_mask().to(device)))

Handling complex model signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The previous section describes how we can trace a model. In some cases it can be
difficult to provide a working argument tuple to |torch.jit.trace|_. Suppose
that we would like to trace a decoder with an attention mask and positions, but
without using a cache. In the
:py:class:`~curated_transformers.models.module.DecoderModule` API, the ``cache``
argument is interspersed between the ``attention_mask`` and ``positions``
arguments. This turns out to be problematic, because we cannot pass ``None``
arguments to the |torch.jit.trace|_ function. |torch.jit.trace|_ provides an
``example_kwarg_inputs`` argument to pass argument by keyword. Unfortulately, we
have found that this mechanism often skips over arguments.

In these cases, we recommend to make a simple wrapper around a model that only
has the desired arguments. For instance, in the case at hand you could define a
class ``DecoderWithPositions``:

.. code-block:: python

   class DecoderWithPositions(Module):
       def __init__(self, decoder: DecoderModule):
           super().__init__()
           self.inner = decoder

       def forward(self, input_ids: Tensor, positions: Tensor):
           return self.inner.forward(input_ids=input_ids, positions=positions)

You can then wrap a decoder with this class and trace it using the two mandatory
arguments.

.. |torch.jit.trace| replace:: ``torch.jit.trace``
.. _torch.jit.trace: https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace
