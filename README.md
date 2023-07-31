<img src="docs/source/logo.png" width="100" align="right"/>

# Curated Transformers

[![Documentation Status](https://readthedocs.org/projects/button/badge/?version=latest)](https://curated-transformers.readthedocs.io/en/latest/?badge=latest)
[![pypi Version](https://img.shields.io/pypi/v/curated-transformers.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/curated-transformers/)

**State-of-the-art transformers, brick by brick**

Curated Transformers is a transformer library for PyTorch. It provides
state-of-the-art models that are composed from a set of reusable
components. The stand-out features of Curated Transformer are:

- ⚡️ Supports state-of-the art transformer models, including LLMs such
  as Falcon, LLaMA, and Dolly v2.
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

- GPT-NeoX
- LLaMA 1/2
- Falcon

Generator wrappers:

- Dolly v2
- Falcon
- LLaMA 1/2

All types of models can be loaded from Huggingface Hub.

spaCy integration for curated transformers is provided by the
[`spacy-curated-transformers`](https://github.com/explosion/spacy-curated-transformers)
package.

## ⚠️ Warning: Tech Preview

Curated Transformers 0.9.x is a tech preview, we will release Curated Transformers
1.0.0 with a stable API and semver guarantees over the coming weeks.

## ⏳ Install

```bash
pip install curated-transformers
```

## 🏃‍♀️ Usage Example

```python-console
>>> from curated_transformers.generation import AutoGenerator, GreedyGeneratorConfig
>>> generator = AutoGenerator.from_hf_hub(name="tiiuae/falcon-7b-instruct", device="cuda:0")
>>> generator(["What is Python in one sentence?", "What is Rust in one sentence?"], GreedyGeneratorConfig())
['Python is a high-level programming language that is easy to learn and widely used for web development, data analysis, and automation.',
 'Rust is a programming language that is designed to be a safe, concurrent, and efficient replacement for C++.']
```

You can find more [usage examples](https://curated-transformers.readthedocs.io/en/latest/usage.html)
in the documentation. You can also find example programs that use Curated Transformers in the
[`examples`](examples/) directory.

## 📚 Documentation

You can read more about how to use Curated Transformers here:

- [Overview](https://curated-transformers.readthedocs.io/en/v0.9.x/) ([Development](https://curated-transformers.readthedocs.io/en/latest/))
- [Usage](https://curated-transformers.readthedocs.io/en/v0.9.x/usage.html) ([Development](https://curated-transformers.readthedocs.io/en/latest/usage.html))
- [API](https://curated-transformers.readthedocs.io/en/v0.9.x/api.html) ([Development](https://curated-transformers.readthedocs.io/en/latest/api.html))

## 🗜️ Quantization

`curated-transformers` supports dynamic 8-bit and 4-bit quantization of models by leveraging the [`bitsandbytes` library](https://github.com/TimDettmers/bitsandbytes).
