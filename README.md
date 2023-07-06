# 🤖 Curated transformers

This Python package provides a curated set of PyTorch transformer models,
composed of reusable modules.

Supported encoder-only models:

- ALBERT
- BERT
- CamemBERT
- RoBERTa
- XLM-RoBERTa

Supported decoder-only models:

- GPT-NeoX
- Refined Web Model (Falcon)

Generator wrappers:

- Dolly v2
- Falcon

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

`curated-transformers` implements dynamic 8-bit and 4-bit quantization of models by leveraging the [`bitsandbytes` library](https://github.com/TimDettmers/bitsandbytes).

### Installation

`curated-transformers` requires functionality that isn't currently present in `bitsandbytes` (as of `v0.39.1`). While the said functionality is optional, we recommend users
to build the library from source by cloning the [following branch](https://github.com/shadeMe/bitsandbytes/tree/linear-layer-device) of our fork. Installation instructions
can be found [here](https://github.com/shadeMe/bitsandbytes/blob/linear-layer-device/compile_from_source.md).

Users can still use the quantization feature without building the `bitsandbytes` library from our fork. However, this will result in increased CPU memory usage during the model
initialization phase. The extra overhead amounts to about 2x the memory required for each model parameter, but this value is not cumulative as parameters are deserialized one-by-one.
GPU memory usage remains unaffected.
