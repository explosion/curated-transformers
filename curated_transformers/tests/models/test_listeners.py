import pytest
from spacy import util
from spacy.training import Example
from thinc.api import Config, reduce_mean

from curated_transformers.models.listeners import build_transformer_layers_listener_v1
from curated_transformers._compat import has_hf_transformers

cfg_string_transformer_layers_listener = """
    # TransformerLayersListener

    [nlp]
    lang = "en"
    pipeline = ["transformer"]

    [components]

    [components.transformer]
    factory = "curated_transformer"
    all_layer_outputs = True

    [components.transformer.model]
    @architectures = "curated-transformers.BertTransformer.v1"
    vocab_size = 28996
    num_hidden_layers = 2
    hidden_width = 60
    piece_encoder = {"@architectures":"curated-transformers.BertWordpieceEncoder.v1"}
    with_spans = {"@architectures":"curated-transformers.WithStridedSpans.v1"}

    [initialize]

    [initialize.components]

    [initialize.components.transformer]

    [initialize.components.transformer.piecer_loader]
    @model_loaders = "curated-transformers.HFPieceEncoderLoader.v1"
    name = "bert-base-cased"
"""


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_transformer_layers_listener():
    config = Config().from_str(cfg_string_transformer_layers_listener)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)

    transformer = nlp.get_pipe("transformer")
    listener = build_transformer_layers_listener_v1(
        layers=2, width=60, pooling=reduce_mean()
    )
    transformer.add_listener(listener, "test")

    docs = [
        nlp.make_doc("Let's test a transformer."),
        nlp.make_doc("Can it handle more than one doc?"),
    ]
    examples = [Example.from_dict(doc, {}) for doc in docs]

    nlp.initialize(lambda: examples)

    # Check prediction.
    transformer.pipe(docs)
    Y = listener.predict(docs)

    hidden_size = 60
    layers = 3  # embed + hidden layers

    assert len(Y) == 2
    assert len(Y[0]) == layers
    assert len(Y[1]) == layers
    assert Y[0][-1].shape == (len(docs[0]), hidden_size)
    assert Y[1][-1].shape == (len(docs[1]), hidden_size)

    # Check update.

    transformer.update(examples)
    Y, backprop = listener.begin_update(docs)
    assert len(Y) == 2
    assert len(Y[0]) == layers
    assert len(Y[1]) == layers
    assert Y[0][-1].shape == (len(docs[0]), hidden_size)
    assert Y[1][-1].shape == (len(docs[1]), hidden_size)

    dY_doc0 = listener.ops.alloc2f(len(docs[0]), 60)
    dY_doc1 = listener.ops.alloc2f(len(docs[1]), 60)

    backprop([layers * [dY_doc0], layers * [dY_doc1]])
