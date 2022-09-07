from numpy.testing import assert_array_equal
from spacy import Config, util
from spacy.training import Example
from thinc.backends import get_current_ops


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
    @architectures = "spacy-experimental.LastTransformerLayerListener.v1"
    width = 768
    pooling = {"@layers":"reduce_mean.v1"}

    [components.tok2vec]
    factory = "experimental_transformer"

    [components.tok2vec.model]
    @architectures = "spacy-experimental.XLMRTransformer.v1"
    hf_model_name = "xlm-roberta-base"

    [components.tok2vec.model.with_spans]
    @architectures = "spacy-experimental.WithStridedSpans.v1"
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
