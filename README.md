<img src="docs/source/logo.png" width="100" align="right"/>

# Curated Transformers

[![Documentation Status](https://readthedocs.org/projects/button/badge/?version=latest)](https://curated-transformers.readthedocs.io/en/latest/?badge=latest)
[![pypi Version](https://img.shields.io/pypi/v/curated-transformers.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/curated-transformers/)

**State-of-the-art transformers, brick by brick**

Curated Transformers is a transformer library for PyTorch. It provides
state-of-the-art models that are composed from a set of reusable
components. The stand-out features of Curated Transformer are:

- ⚡️ Supports state-of-the art transformer models, including LLMs such
  as Falcon, Llama, and Dolly v2.
- 👩‍🎨 Each model is composed from a set of reusable building blocks,
  providing many benefits:
  - Implementing a feature or bugfix benefits all models. For example,
    all models support 4/8-bit inference through the
    [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) library
    and each model can use the PyTorch `meta` device to avoid unnecessary
    allocations and initialization.
  - Adding new models to the library is low-effort.
  - Do you want to try a new transformer architecture? A BERT encoder
    with rotary embeddings? You can make it in a pinch.
- 💎 Consistent type annotations of all public APIs:
  - Get great coding support from your IDE.
  - Integrates well with your existing type-checked code.
- 🎓 Great for education, because the building blocks are easy to study.
- 📦 Minimal dependencies.

Curated Transformers has been production-tested by [Explosion](http://explosion.ai/)
and will be used as the default transformer implementation in spaCy 3.7.

## 🧰 Supported Model Architectures

Supported encoder-only models:

- ALBERT
- BERT
- CamemBERT
- RoBERTa
- XLM-RoBERTa

Supported decoder-only models:

- Falcon
- GPT-NeoX
- Llama 1/2
- MPT

Generator wrappers:

- Dolly v2
- Falcon
- Llama 1/2
- MPT

All types of models can be loaded from Huggingface Hub.

spaCy integration for curated transformers is provided by the
[`spacy-curated-transformers`](https://github.com/explosion/spacy-curated-transformers)
package.

## ⏳ Install

```bash
pip install curated-transformers
```

### CUDA support

The default Linux build of PyTorch is built with CUDA 11.7 support. You should
explicitly install a CUDA build in the following cases:

- If you want to use Curated Transformers on Windows.
- If you want to use Curated Transformers on Linux with Ada-generation GPUs.
  The standard PyTorch build supports Ada GPUs, but you can get considerable
  performance improvements by installing PyTorch with CUDA 11.8 support.

In both cases, you can install PyTorch with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 🏃‍♀️ Usage Example

```python-console
>>> import torch
>>> from curated_transformers.generation import AutoGenerator, GreedyGeneratorConfig
>>> generator = AutoGenerator.from_hf_hub(name="tiiuae/falcon-7b-instruct", device=torch.device("cuda"))
>>> generator(["What is Python in one sentence?", "What is Rust in one sentence?"], GreedyGeneratorConfig())
['Python is a high-level programming language that is easy to learn and widely used for web development, data analysis, and automation.',
 'Rust is a programming language that is designed to be a safe, concurrent, and efficient replacement for C++.']
```

You can find more [usage examples](https://curated-transformers.readthedocs.io/en/latest/usage.html)
in the documentation. You can also find example programs that use Curated Transformers in the
[`examples`](examples/) directory.

## 📚 Documentation

You can read more about how to use Curated Transformers here:

- [Overview](https://curated-transformers.readthedocs.io/en/v1.2.x/) ([Development](https://curated-transformers.readthedocs.io/en/latest/))
- [Usage](https://curated-transformers.readthedocs.io/en/v1.2.x/usage.html) ([Development](https://curated-transformers.readthedocs.io/en/latest/usage.html))
- [API](https://curated-transformers.readthedocs.io/en/v1.2.x/api.html) ([Development](https://curated-transformers.readthedocs.io/en/latest/api.html))

## 🗜️ Quantization

`curated-transformers` supports dynamic 8-bit and 4-bit quantization of models by leveraging the [`bitsandbytes` library](https://github.com/TimDettmers/bitsandbytes).

Use the quantization variant to automatically install the necessary dependencies:

```bash
pip install curated-transformers[quantization]
```
