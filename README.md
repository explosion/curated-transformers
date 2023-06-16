# 🤖 Curated transformers

This Python package provides a curated set of PyTorch transformer models,
composed of reusable modules. Curated transformers currently supports the
following model types:

- ALBERT
- BERT
- CamemBERT
- RoBERTa
- XLM-RoBERTa

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

Since `curated-transformers` requires functionality that isn't currently present in `bitsandbytes` (as of `v0.39.0`), one needs to install the library
from source by cloning the [following branch](https://github.com/shadeMe/bitsandbytes/tree/linear-layer-device) of our fork. Installation instructions
can be found [here](https://github.com/shadeMe/bitsandbytes/blob/linear-layer-device/compile_from_source.md).
