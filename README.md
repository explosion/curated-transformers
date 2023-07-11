# 🤖 Curated Transformers

**Only one attention layer in eight models**

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
  - Get great feedback from your IDE.
  - Integrates well with your existing type-checked code.
- 🎓 Great for education, because the building blocks are easy to study.
- 📦 Minimal dependencies.

Curated Transformers has been production-tested by [Explosion](http://explosion.ai/) 
and will be used as the default transformer implementation in spaCy 3.7.

## Supported model architectures

Supported encoder-only models:

- ALBERT
- BERT
- CamemBERT
- RoBERTa
- XLM-RoBERTa

Supported decoder-only models:

- GPT-NeoX
- LLaMA
- Refined Web Model (Falcon)

Generator wrappers:

- Dolly v2
- Falcon

All types of models can be loaded from Huggingface Hub.

spaCy integration for curated transformers is provided by the
[`spacy-curated-transformers`](https://github.com/explosion/spacy-curated-transformers)
package.

## ⚠️ Warning: experimental package

This package is experimental and it is possible that models and APIs will
change in incompatible ways.

## ⏳ Install

```bash
pip install curated-transformers
```

## Quantization

`curated-transformers` supports dynamic 8-bit and 4-bit quantization of models by leveraging the [`bitsandbytes` library](https://github.com/TimDettmers/bitsandbytes).
