from spacy.cli import app
from typer.testing import CliRunner

import curated_transformers.cli.quantize
import curated_transformers.cli.debug_pieces


def test_quantize():
    result = CliRunner().invoke(app, ["quantize-transformer", "--help"])
    assert result.exit_code == 0


def test_debug_pieces():
    result = CliRunner().invoke(app, ["debug", "pieces", "--help"])
    assert result.exit_code == 0
