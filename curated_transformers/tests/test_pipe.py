from typing import Dict, Any
from functools import partial
import pytest
import spacy
from spacy import Config, util
from spacy.training import Example
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from spacy.language import Language
from spacy.util import registry as spacy_registry
import torch

from curated_transformers.models.transformer_model import (
    build_camembert_transformer_model_v1,
    build_xlmr_transformer_model_v1,
    build_bert_transformer_model_v1,
    build_roberta_transformer_model_v1,
)
from curated_transformers.models.with_strided_spans import (
    build_with_strided_spans_v1,
)
from curated_transformers.models.pytorch.attention import AttentionMask
from curated_transformers.models.hf_loader import build_hf_encoder_loader_v1
from curated_transformers.tokenization.hf_loader import build_hf_piece_encoder_loader_v1
from curated_transformers.pipe import make_transformer
from curated_transformers.util import create_gradual_transformer_unfreezing
from curated_transformers._compat import has_hf_transformers, transformers

from .util import make_tempdir


cfg_string_last_layer_listener = """
    # LastTransformerLayerListener

    [nlp]
    lang = "en"
    pipeline = ["transformer","tagger"]

    [components]

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v2"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "curated-transformers.LastTransformerLayerListener.v1"
    width = ${components.transformer.model.hidden_width}
    pooling = {"@layers":"reduce_mean.v1"}

    [components.transformer]
    factory = "curated_transformer"
    all_layer_outputs = False

    [components.transformer.model]
    @architectures = "curated-transformers.BertTransformer.v1"
    vocab_size = 28996
    num_hidden_layers = 1
    hidden_width = 60

    [components.transformer.model.with_spans]
    @architectures = "curated-transformers.WithStridedSpans.v1"

    [initialize]

    [initialize.components]

    [initialize.components.transformer]

    [initialize.components.transformer.piecer_loader]
    @model_loaders = "curated-transformers.HFPieceEncoderLoader.v1"
    name = "bert-base-cased"
"""

cfg_string_scalar_weighting_layer_listener = """
    # ScalarWeightingListener

    [nlp]
    lang = "en"
    pipeline = ["transformer","tagger"]

    [components]

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v2"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "curated-transformers.ScalarWeightingListener.v1"
    width = ${components.transformer.model.hidden_width}
    pooling = {"@layers":"reduce_mean.v1"}

    [components.tagger.model.tok2vec.weighting]
    @architectures = "curated-transformers.ScalarWeight.v1"
    num_layers = ${components.transformer.model.num_hidden_layers}

    [components.transformer]
    factory = "curated_transformer"
    all_layer_outputs = True

    [components.transformer.model]
    @architectures = "curated-transformers.BertTransformer.v1"
    vocab_size = 28996
    num_hidden_layers = 1
    hidden_width = 60

    [components.transformer.model.with_spans]
    @architectures = "curated-transformers.WithStridedSpans.v1"

    [initialize]

    [initialize.components]

    [initialize.components.transformer]

    [initialize.components.transformer.piecer_loader]
    @model_loaders = "curated-transformers.HFPieceEncoderLoader.v1"
    name = "bert-base-cased"
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


def create_and_train_tagger(cfg_string):
    config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)
    tagger = nlp.get_pipe("tagger")

    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)

    optimizer = nlp.initialize(lambda: train_examples)

    for _ in range(10):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    return nlp


def evaluate_tagger_on_train_data(model):
    docs = list(model.pipe(["Eat blue ham", "I like green eggs"]))
    assert [t.tag_ for t in docs[0]] == ["V", "J", "N"]
    assert [t.tag_ for t in docs[1]] == ["N", "V", "J", "N"]


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize(
    "cfg_string",
    [cfg_string_last_layer_listener, cfg_string_scalar_weighting_layer_listener],
)
def test_tagger(cfg_string):
    model = create_and_train_tagger(cfg_string)
    evaluate_tagger_on_train_data(model)


def _hf_tokenize_per_token(tokenizer, docs, *, roberta=False):
    if roberta:
        hf_encoding = [
            tokenizer(
                [
                    doc[idx - 1].whitespace_ + token.text if idx > 0 else token.text
                    for idx, token in enumerate(doc)
                ]
            )
            for doc in docs
        ]
    else:
        hf_encoding = [tokenizer([token.text for token in doc]) for doc in docs]
    ids = []
    lens = []
    bos_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.cls_token_id
    )
    eos_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.sep_token_id
    )
    for i in range(len(hf_encoding)):
        doc_ids = [id for e in hf_encoding[i]["input_ids"] for id in e[1:-1]]
        ids.append([bos_id] + doc_ids + [eos_id])
        lens.append(len(ids[-1]))

    torch_ids = torch.full(
        (len(ids), max(lens)), tokenizer.pad_token_id, dtype=torch.int
    )
    for i in range(len(ids)):
        torch_ids[i][: len(ids[i])] = torch.tensor(ids[i])

    attention_mask = torch_ids.ne(tokenizer.pad_token_id)

    return torch_ids, attention_mask, lens


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_bert_transformer_pipe_against_hf():
    nlp = spacy.blank("en")
    model = build_bert_transformer_model_v1(
        with_spans=build_with_strided_spans_v1(),
        vocab_size=28996,
    )
    model.get_ref("transformer").init = build_hf_encoder_loader_v1(
        name="bert-base-cased"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="bert-base-cased"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    hf_model = transformers.AutoModel.from_pretrained("bert-base-cased")

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
            hf_doc_encoding[:encoding_len][1:-1],
            doc._.trf_data.last_hidden_layer_state.dataXd,
        )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_camembert_transformer_pipe_against_hf():
    nlp = spacy.blank("fr")
    model = build_camembert_transformer_model_v1(
        with_spans=build_with_strided_spans_v1(),
        vocab_size=32005,
    )
    model.get_ref("transformer").init = build_hf_encoder_loader_v1(
        name="camembert-base"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="camembert-base"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("camembert-base")
    hf_model = transformers.AutoModel.from_pretrained("camembert-base")

    docs = [
        nlp.make_doc("J'ai vu une fille avec un télescope."),
        nlp.make_doc("Aujourd'hui, nous allons manger un poké bowl."),
    ]

    hf_ids, attention_mask, lens = _hf_tokenize_per_token(hf_tokenizer, docs)
    hf_encoding = hf_model(hf_ids, attention_mask=attention_mask)
    docs = list(pipe.pipe(docs))

    for doc, hf_doc_encoding, encoding_len in zip(
        docs, hf_encoding.last_hidden_state, lens
    ):
        torch.testing.assert_allclose(
            hf_doc_encoding[:encoding_len][1:-1],
            doc._.trf_data.last_hidden_layer_state.dataXd,
        )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_roberta_transformer_pipe_against_hf():
    nlp = spacy.blank("en")
    model = build_roberta_transformer_model_v1(
        with_spans=build_with_strided_spans_v1(),
        vocab_size=50265,
    )
    model.get_ref("transformer").init = build_hf_encoder_loader_v1(name="roberta-base")
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="roberta-base"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    hf_model = transformers.AutoModel.from_pretrained("roberta-base")

    docs = [
        nlp.make_doc("I saw a girl with a telescope."),
        nlp.make_doc("Today we will eat poké bowl."),
    ]

    hf_ids, attention_mask, lens = _hf_tokenize_per_token(
        hf_tokenizer, docs, roberta=True
    )
    hf_encoding = hf_model(hf_ids, attention_mask=attention_mask)
    docs = list(pipe.pipe(docs))

    for doc, hf_doc_encoding, encoding_len in zip(
        docs, hf_encoding.last_hidden_state, lens
    ):
        torch.testing.assert_allclose(
            hf_doc_encoding[:encoding_len][1:-1],
            doc._.trf_data.last_hidden_layer_state.dataXd,
        )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_xlmr_transformer_pipe_against_hf():
    nlp = spacy.blank("en")
    model = build_xlmr_transformer_model_v1(
        with_spans=build_with_strided_spans_v1(),
        vocab_size=250002,
    )
    model.get_ref("transformer").init = build_hf_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.initialize()
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
            hf_doc_encoding[:encoding_len][1:-1],
            doc._.trf_data.last_hidden_layer_state.dataXd,
        )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_frozen_transformer_pipe():
    config = Config().from_str(cfg_string_scalar_weighting_layer_listener)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)
    tagger = nlp.get_pipe("tagger")
    transformer = nlp.get_pipe("transformer")
    transformer.frozen = True

    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)

    optimizer = nlp.initialize(lambda: train_examples)

    def get_transformer_params_sorted():
        params = transformer.model.get_ref("transformer").shims[0]._model.state_dict()
        return list(sorted(params.items()))

    transformer_init_params = [
        (k, v.clone()) for k, v in get_transformer_params_sorted()
    ]

    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    transformer_trained_params = get_transformer_params_sorted()
    for ((old_param, old_vec), (new_param, new_vec)) in zip(
        transformer_init_params, transformer_trained_params
    ):
        assert old_param == new_param
        torch.testing.assert_allclose(old_vec, new_vec)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_transformer_pipe_outputs():
    nlp = spacy.blank("en")
    model = build_xlmr_transformer_model_v1(
        with_spans=build_with_strided_spans_v1(),
        vocab_size=250002,
    )
    model.get_ref("transformer").init = build_hf_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model, all_layer_outputs=False)

    docs = [
        nlp.make_doc("I saw a girl with a telescope."),
        nlp.make_doc("Today we will eat poké bowl."),
    ]
    docs = list(pipe.pipe(docs))
    assert all([doc._.trf_data.last_layer_only for doc in docs]) == True
    assert all([len(doc._.trf_data.all_outputs) == 1 for doc in docs]) == True

    pipe = make_transformer(nlp, "transformer", model, all_layer_outputs=True)
    docs = list(pipe.pipe(docs))
    assert all([not doc._.trf_data.last_layer_only for doc in docs]) == True
    assert all([len(doc._.trf_data.all_outputs) == 12 + 1 for doc in docs]) == True


cfg_string_gradual_unfreezing = (
    cfg_string_last_layer_listener
    + """
    [corpora]

    [corpora.train]
    @readers = "spacy.Corpus.v1"
    path = "toy-en-corpus.spacy"
    max_length = 500
    gold_preproc = false
    limit = 0

    [corpora.dev]
    @readers = "spacy.Corpus.v1"
    path = "toy-en-corpus.spacy"
    max_length = 500
    gold_preproc = false
    limit = 0

    [training]
    train_corpus = "corpora.train"
    dev_corpus = "corpora.dev"
    seed = 1
    gpu_allocator = "pytorch"
    dropout = 0.1
    accumulate_gradient = 3
    patience = 5000
    max_epochs = 1
    max_steps = 6
    eval_frequency = 10

    [training.batcher]
    @batchers = "spacy.batch_by_padded.v1"
    discard_oversize = False
    get_length = null
    size = 1
    buffer = 256
"""
)


@pytest.mark.slow
@pytest.mark.xfail(reason="fails until spaCy supports after_update callbacks")
@pytest.mark.skipif(not has_hf_transformers, reason="requires 🤗 transformers")
def test_gradual_transformer_unfreezing(test_dir):
    def wrapped_callback():
        def inner(
            nlp: Language,
            args: Dict[str, Any],
            gradual_unfreezing_callback,
            unfreeze_step,
        ):
            current_step = args["step"]
            gradual_unfreezing_callback(nlp, args)

            transformer = nlp.get_pipe("transformer")
            if current_step < unfreeze_step:
                assert transformer.frozen == True
            else:
                assert transformer.frozen == False

        return partial(
            inner,
            gradual_unfreezing_callback=create_gradual_transformer_unfreezing({"*": 3}),
            unfreeze_step=3,
        )

    spacy_registry.callbacks.register(
        "test_gradual_unfreezing_callback",
        func=wrapped_callback,
    )

    config = Config().from_str(cfg_string_gradual_unfreezing, interpolate=False)
    config["corpora"]["train"]["path"] = str(test_dir / "toy-en-corpus.spacy")
    config["corpora"]["dev"]["path"] = str(test_dir / "toy-en-corpus.spacy")
    config["training"]["before_update"] = {
        "@callbacks": "test_gradual_unfreezing_callback"
    }
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)

    nlp = init_nlp(config)
    train(nlp)
    assert nlp.get_pipe("transformer").frozen == False

    with pytest.raises(ValueError):
        create_gradual_transformer_unfreezing({"transformer": 5, "*": 4})
