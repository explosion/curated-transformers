Usage
=====

.. _installation:

Installation
------------

To use Curated Transformers, first install it using ``pip``:

.. code-block:: console

   (.venv) $ pip install curated-transformers

If support for quantization is required, use the quantization variant to automatically install the necessary dependencies:

.. code-block:: console

   (.venv) $ pip install curated-transformers[quantization]

CUDA Support
^^^^^^^^^^^^

The default Linux build of `PyTorch`_ is built with `CUDA`_ 11.7 support. You should
explicitly install a CUDA build in the following cases:

- If you want to use Curated Transformers on Windows.
- If you want to use Curated Transformers on Linux with Ada-generation GPUs.

  The standard PyTorch build supports Ada GPUs, but you can get considerable
  performance improvements by installing PyTorch with CUDA 11.8 support.

In both cases, you can install PyTorch with:

.. code-block:: console

   (.venv) $ pip install torch --index-url https://download.pytorch.org/whl/cu118

.. _PyTorch: https://pytorch.org/
.. _CUDA: https://developer.nvidia.com/cuda-toolkit


Text Generation Using Causal LMs
--------------------------------

Curated Transformers provides infrastructure to perform open-ended text generation using decoder-only causal language models. 
The :py:class:`~curated_transformers.generation.Generator` class wraps a :py:class:`~curated_transformers.models.CausalLMModule`
and its corresponding tokenizer. It provides a generic interface to generate outputs from the wrapped module in an auto-regressive fashion. 
:py:class:`~curated_transformers.generation.GeneratorConfig` specifies the parameters used by the generator such as stopping conditions
and sampling parameters.

The :py:class:`~curated_transformers.generation.AutoGenerator` class can be used to directly load a supported causal
LM model and generate text with it.

.. code-block:: python

      import torch
      from curated_transformers.generation import (
         AutoGenerator,
         GreedyGeneratorConfig,
         SampleGeneratorConfig,
      )

      generator = AutoGenerator.from_hf_hub(
         name="databricks/dolly-v2-3b", device=torch.device("cuda", index=0)
      )

      sample_config = SampleGeneratorConfig(temperature=1.0, top_k=2)
      greedy_config = GreedyGeneratorConfig()

      prompts = [
         "To which David Bowie song do these lyrics belong: \"Oh man, look at those cavemen go! It's the freakiest show\"?",
         "What is spaCy?"
      ]
      sample_outputs = generator(prompts, config=sample_config)
      greedy_outputs = generator(prompts, config=greedy_config)

      print(f"Sampling outputs: {sample_outputs}")
      print(f"Greedy outputs: {greedy_outputs}")


For more information about the different configs and generators supported by Curated Transformers, see :ref:`generation`.


Loading a Model
---------------

Curated Transformers allows users to easily load model weights from the `Hugging Face Model Hub`_. All models 
provide a ``from_hf_hub`` method that allows directly loading pre-trained model parameters from Hugging Face 
Model Hub.

.. _Hugging Face Model Hub: https://huggingface.co/models

.. code-block:: python

   import torch
   from curated_transformers.models import BERTEncoder
   from curated_transformers.models import GPTNeoXDecoder

   encoder = BERTEncoder.from_hf_hub(
      name="bert-base-uncased",
      revision="main",
      device=torch.device("cuda", index=0),
   )

   decoder = GPTNeoXDecoder.from_hf_hub(name="databricks/dolly-v2-3b", revision="main")


The :py:class:`~curated_transformers.models.AutoEncoder`, :py:class:`~curated_transformers.models.AutoDecoder`
and :py:class:`~curated_transformers.models.AutoCausalLM` classes can be used to automatically infer the model architecture.

.. code-block:: python

   from curated_transformers.models import (
      AutoCausalLM,
      AutoDecoder,
      AutoEncoder,
   )

   encoder = AutoEncoder.from_hf_hub(
      name="bert-base-uncased",
      revision="main",
   )

   decoder = AutoDecoder.from_hf_hub(name="databricks/dolly-v2-3b", revision="main")

   lm = AutoCausalLM.from_hf_hub(name="databricks/dolly-v2-3b", revision="main")


Quantization
------------

Curated Transformers implements dynamic 8-bit and 4-bit quantization of models by leveraging the `bitsandbytes`_ library.
When loading models using the ``from_hf_hub`` method, an optional :py:class:`~curated_transformers.quantization.bnb.BitsAndBytesConfig`
instance can be passed to the method to opt into dynamic quantization of model parameters. Quantization requires the model to be
loaded to a CUDA GPU by additionally passing the ``device`` argument to the method.

.. _bitsandbytes: https://github.com/TimDettmers/bitsandbytes

.. code-block:: python

   import torch
   from curated_transformers.generation import AutoGenerator
   from curated_transformers.quantization.bnb import BitsAndBytesConfig, Dtype4Bit

   generator_8bit = AutoGenerator.from_hf_hub(
      name="databricks/dolly-v2-3b",
      device=torch.device("cuda", index=0),
      quantization_config=BitsAndBytesConfig.for_8bit(
         outlier_threshold=6.0, finetunable=False
      ),
   )

   generator_4bit = AutoGenerator.from_hf_hub(
      name="databricks/dolly-v2-3b",
      device=torch.device("cuda", index=0),
      quantization_config=BitsAndBytesConfig.for_4bit(
         quantization_dtype=Dtype4Bit.FP4,
         compute_dtype=torch.bfloat16,
         double_quantization=True,
      ),
   )


Loading a Tokenizer
-------------------

To train or run inference on the models, one has to tokenize the inputs with a compatible tokenizer. Curated Transformers supports 
tokenizers implemented by the `Hugging Face tokenizers`_ library and certain model-specific tokenizers that are implemented 
using the `Curated Tokenizers`_ library. The :py:class:`~curated_transformers.tokenizers.Tokenizer` class encapsulates the
former and the :py:class:`~curated_transformers.tokenizers.legacy.LegacyTokenizer` class the latter.

In both cases, one can use the :py:class:`~curated_transformers.tokenizers.AutoTokenizer` class to automatically
infer the correct tokenizer type and construct a Curated Transformers tokenizer that implements the :py:class:`~curated_transformers.tokenizers.TokenizerBase`
interface.

.. code-block:: python

   from curated_transformers.tokenizers import AutoTokenizer

   tokenizer = AutoTokenizer.from_hf_hub(
      name="bert-base-uncased",
      revision="main",
   )

.. _Hugging Face tokenizers: https://github.com/huggingface/tokenizers
.. _Curated Tokenizers: https://github.com/explosion/curated-tokenizers


Text Encoding
-------------

.. note::
   Currently, Curated Transformers only supports inference with models.

In addition to text generation, one can also run inference on the inputs to produce their dense representations.

.. code-block:: python

   import torch
   from curated_transformers.models import AutoEncoder
   from curated_transformers.tokenizers import AutoTokenizer

   device = torch.device("cpu")
   encoder = AutoEncoder.from_hf_hub(
      name="bert-base-uncased", revision="main", device=device
   )
   # Set module state to evaluation mode.
   encoder.eval()

   tokenizer = AutoTokenizer.from_hf_hub(
      name="bert-base-uncased",
      revision="main",
   )

   input_pieces = tokenizer(
      [
         "Straight jacket fitting a little too tight",
         "Space shuttle, snail shell, merry go round, conveyor belt!",
      ]
   )

   # Don't allocate gradients since we're only running inference.
   with torch.no_grad():
      ids = input_pieces.padded_tensor(pad_left=True, device=device)
      attention_mask = input_pieces.attention_mask(device=device)
      model_output = encoder(ids, attention_mask)

   # [batch, seq_len, width]
   last_hidden_repr = model_output.last_hidden_layer_state


The :py:class:`~curated_transformers.models.ModelOutput` instance returned by the encoder contains all of
transformer's outputs, i.e., the hidden representations of all transformer layers and the output of the embedding
layer. Decoder models (:py:class:`~curated_transformers.models.DecoderModule`) and causal language models
(:py:class:`~curated_transformers.models.CausalLMModule`) produce additional outputs such as the key-value
cache used during attention calculation (:py:class:`~curated_transformers.models.ModelOutputWithCache`) and
logits (:py:class:`~curated_transformers.models.CausalLMOutputWithCache`).
