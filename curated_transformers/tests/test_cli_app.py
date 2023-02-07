from spacy.cli import app
from typer.testing import CliRunner

import curated_transformers.cli.quantize


def test_quantize():
    result = CliRunner().invoke(app, ["quantize-transformer", "--help"])
    assert result.exit_code == 0
