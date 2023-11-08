from pathlib import Path

import pytest
import torch

TORCH_DEVICES = [torch.device("cpu")]
GPU_TESTS_ENABLED = False


def pytest_addoption(parser):
    try:
        parser.addoption("--slow", action="store_true", help="include slow tests")
        parser.addoption(
            "--upload-tests",
            action="store_true",
            help="include tests that upload test artifacts to remote repos",
        )
    # Options are already added, e.g. if conftest is copied in a build pipeline
    # and runs twice
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: include slow tests")
    config.addinivalue_line(
        "markers", "upload-tests: tests that upload test artifacts to remote repos"
    )


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
    for opt in ["upload_tests", "slow"]:
        if opt in item.keywords and not getopt(opt.replace("_", "-")):
            pytest.skip(f"need --{opt.replace('_', '-')} option to run")


@pytest.fixture
def test_dir(request):
    return Path(request.fspath).parent


@pytest.fixture
def french_sample_texts():
    return [
        "J'ai vu une fille avec un télescope.",
        "Aujourd'hui, nous allons manger un poké bowl.",
    ]


@pytest.fixture
def short_sample_texts():
    return [
        "I saw a girl with a telescope.",
        "Today we will eat poké bowl, lots of it!",
        "Tokens which are unknown inペ mostで latinが alphabet際 vocabularies.",
    ]


@pytest.fixture
def sample_texts():
    # Two short Wikipedia fragments from:
    # https://en.wikipedia.org/wiki/Kinesis_(keyboard)#Contoured_/_Advantage
    # https://en.wikipedia.org/wiki/Doom_(1993_video_game)#Engine
    return [
        "The original Model 100, released in 1992, featured a single-piece "
        "contoured design similar to the Maltron keyboard, with the keys laid "
        "out in a traditional QWERTY arrangement, separated into two clusters "
        "for the left and right hands.[2] A 1993 article in PC Magazine "
        "described the US$690 (equivalent to $1,300 in 2021) keyboard's "
        'arrangement as having "the alphabet keys in precisely vertical '
        "(not diagonal) columns in two concave depressions. The Kinesis "
        "Keyboard also puts the Backspace, Delete, Enter, Space, Ctrl, Alt, "
        "Home, End, Page Up, and Page Down keys under your thumbs in the "
        'middle".[23]',
        "Doom was programmed largely in the ANSI C programming language, with "
        "a few elements in assembly language. Development was done on NeXT "
        "computers running the NeXTSTEP operating system.[35] The data used by "
        "the game engine, including level designs and graphics files, are "
        'stored in WAD files, short for "Where\'s All the Data?".',
    ]
