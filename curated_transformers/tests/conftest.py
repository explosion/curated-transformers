from pathlib import Path
import pytest
import torch


TORCH_DEVICES = [torch.device("cpu")]


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
