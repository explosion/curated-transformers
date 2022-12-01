# 🤖 Curated transformers

This Python package provides a curated set of transformer models for spaCy. It
is focused on deep integration into spaCy and will support deployment-focused
features such as distillation and quantization. Curated transformers currently
supports the following model types:

- ALBERT
- BERT
- CamemBERT
- RoBERTa
- XLM-RoBERTa

Supporting a wide variety of transformer models is a non-goal. If you want
to use another type of model, use
[`spacy-transformers`](https://github.com/explosion/spacy-transformers), which
allows you to use [Hugging Face
`transformers`](https://github.com/huggingface/transformers) models with spaCy.

## ⚠️ Warning: experimental package

This package is experimental and it is possible that the models will still
change in incompatible ways.

## ⏳ Install

```bash
pip install git+https://github.com/explosion/curated-transformers.git
```

## 🚀 Quickstart

An example project is provided in the [`project`](project) directory.
