from pathlib import Path
import pytest
import spacy

from curated_transformers.tokenization.wordpiece_encoder import build_wordpiece_encoder
from curated_transformers.util import registry


def pytest_addoption(parser):
    try:
        parser.addoption("--slow", action="store_true", help="include slow tests")
    # Options are already added, e.g. if conftest is copied in a build pipeline
    # and runs twice
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: include slow tests")


def pytest_runtest_setup(item):
    def getopt(opt):
        # When using 'pytest --pyargs spacy' to test an installed copy of
        # spacy, pytest skips running our pytest_addoption() hook. Later, when
        # we call getoption(), pytest raises an error, because it doesn't
        # recognize the option we're asking about. To avoid this, we need to
        # pass a default value. We default to False, i.e., we act like all the
        # options weren't given.
        return item.config.getoption(f"--{opt}", False)

    # Integration of boolean flags
    for opt in ["slow"]:
        if opt in item.keywords and not getopt(opt):
            pytest.skip(f"need --{opt} option to run")


@pytest.fixture
def test_dir(request):
    print(request.fspath)
    return Path(request.fspath).parent


@pytest.fixture
def sample_docs_with_spaces():
    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl    with a telescope.")
    doc2 = nlp.make_doc("Today we    will eat    poké bowl.  ")
    return [doc1, doc2]


@pytest.fixture
def sample_docs():
    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")
    return [doc1, doc2]


@pytest.fixture
def wordpiece_toy_model_path():
    return Path(__file__).parent / "tokenization" / "toy.wordpieces"


@pytest.fixture
def wordpiece_toy_encoder(wordpiece_toy_model_path):
    encoder = build_wordpiece_encoder()
    encoder.init = registry.model_loaders.get(
        "curated-transformers.WordpieceLoader.v1"
    )(path=wordpiece_toy_model_path)
    encoder.initialize()
    return encoder
