from numpy.testing import assert_array_equal
import pytest
import spacy
from spacy import Config, util
from spacy.training import Example
from thinc.api import NumpyOps
from thinc.backends import get_current_ops
import torch

from curated_transformers.models.transformer_model import (
    build_xlmr_transformer_model_v1,
)
from curated_transformers.models.with_strided_spans import (
    build_with_strided_spans_v1,
)
from curated_transformers.pipe import make_transformer
from curated_transformers._compat import has_hf_transformers, transformers


cfg_string = """
    [nlp]
    lang = "en"
    pipeline = ["tok2vec","tagger"]

    [components]

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v2"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "curated-transformers.LastTransformerLayerListener.v1"
    width = 768
    pooling = {"@layers":"reduce_mean.v1"}

    [components.tok2vec]
    factory = "curated_transformer"

    [components.tok2vec.model]
    @architectures = "curated-transformers.XLMRTransformer.v1"
    hf_model_name = "xlm-roberta-base"

    [components.tok2vec.model.with_spans]
    @architectures = "curated-transformers.WithStridedSpans.v1"
"""

TRAIN_DATA = [
    (
        "I like green eggs",
        {"tags": ["N", "V", "J", "N"], "cats": {"preference": 1.0, "imperative": 0.0}},
    ),
    (
        "Eat blue ham",
        {"tags": ["V", "J", "N"], "cats": {"preference": 0.0, "imperative": 1.0}},
    ),
]


def test_tagger():
    config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)
    tagger = nlp.get_pipe("tagger")

    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)

    optimizer = nlp.initialize(lambda: train_examples)

    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    docs = list(nlp.pipe(["Eat blue ham", "I like green eggs"]))
    assert [t.tag_ for t in docs[0]] == ["V", "J", "N"]
    assert [t.tag_ for t in docs[1]] == ["N", "V", "J", "N"]


def _hf_tokenize_per_token(tokenizer, docs):
    hf_encoding = [tokenizer([token.text for token in doc]) for doc in docs]
    ids = []
    lens = []
    for i in range(len(hf_encoding)):
        doc_ids = [id for e in hf_encoding[i]["input_ids"] for id in e[1:-1]]
        ids.append([tokenizer.bos_token_id] + doc_ids + [tokenizer.eos_token_id])
        lens.append(len(ids[-1]))

    torch_ids = torch.full(
        (len(ids), max(lens)), tokenizer.pad_token_id, dtype=torch.int
    )
    for i in range(len(ids)):
        torch_ids[i][: len(ids[i])] = torch.tensor(ids[i])

    attention_mask = torch_ids.ne(tokenizer.pad_token_id)

    return torch_ids, attention_mask, lens


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_transformer_pipe_against_hf():
    nlp = spacy.blank("en")
    model = build_xlmr_transformer_model_v1(
        with_spans=build_with_strided_spans_v1(), hf_model_name="xlm-roberta-base"
    )
    pipe = make_transformer(nlp, "transformer", model)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    hf_model = transformers.AutoModel.from_pretrained("xlm-roberta-base")

    docs = [
        nlp.make_doc("I saw a girl with a telescope."),
        nlp.make_doc("Today we will eat poké bowl."),
    ]

    hf_ids, attention_mask, lens = _hf_tokenize_per_token(hf_tokenizer, docs)
    hf_encoding = hf_model(hf_ids, attention_mask=attention_mask)
    docs = list(pipe.pipe(docs))

    for doc, hf_doc_encoding, encoding_len in zip(
        docs, hf_encoding.last_hidden_state, lens
    ):
        torch.testing.assert_allclose(
            hf_doc_encoding[:encoding_len][1:-1], doc._.trf_data.dataXd
        )
