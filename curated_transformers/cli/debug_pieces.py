from typing import Any, Dict, List, Optional, cast
import numpy
from pathlib import Path
from spacy import util
from spacy.cli._util import (
    debug_cli,
    import_code,
    parse_config_overrides,
    show_validation_error,
)
from spacy.schemas import ConfigSchemaTraining
from spacy.tokens import Doc
from spacy.util import registry, resolve_dot_names
from typer import Argument as Arg, Context, Option as Opt
from wasabi import Printer, msg

from ..tokenization.types import Tok2PiecesModelT
from ..pipeline.transformer import Transformer


@debug_cli.command(
    "pieces",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def debug_pieces_cli(
    # fmt: off
    ctx: Context,  # This is only used to read additional arguments
    config_path: Path = Arg(..., help="Path to config file", exists=True, allow_dash=True),
    code_path: Optional[Path] = Opt(None, "--code-path", "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    transformer_name: Optional[str] = Opt(None, "--name", "-n", help="Name of the transformer pipe to gather piece statistics for (default: first transformer pipe)."),
    # fmt: on
):
    """
    Analyze word- or sentencepiece statistics.
    DOCS: https://spacy.io/api/cli#debug-pieces
    """
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    debug_pieces(
        config_path,
        config_overrides=overrides,
        transformer_name=transformer_name,
    )


def debug_pieces(
    config_path: Path,
    *,
    config_overrides: Dict[str, Any] = {},
    transformer_name: Optional[str] = None,
):
    msg = Printer()

    with show_validation_error(config_path):
        cfg = util.load_config(config_path, overrides=config_overrides)
        nlp = util.load_model_from_config(cfg, auto_fill=True)
        config = nlp.config.interpolate()
        T = registry.resolve(config["training"], schema=ConfigSchemaTraining)

    dot_names = [T["train_corpus"], T["dev_corpus"]]
    train_corpus, dev_corpus = resolve_dot_names(config, dot_names)

    nlp.initialize(lambda: train_corpus(nlp))

    if transformer_name is None:
        transformers = [
            pipe for _, pipe in nlp.pipeline if isinstance(pipe, Transformer)
        ]
        if not transformers:
            msg.fail("Pipeline does not contain transformer", exits=1)
        transformer_pipe = transformers[0]
    else:
        # We have to dance a bit around that MyPy cannot infer that we are
        # exiting if the invariants don't hold and that get_pipe returns
        # Callable.
        try:
            transformer_pipe_callable = nlp.get_pipe(transformer_name)
        except KeyError:
            transformer_pipe_callable = None
            msg.fail(
                f"Pipeline does not contain a pipe named '{transformer_name}'", exits=1
            )
        if not isinstance(transformer_pipe_callable, Transformer):
            msg.fail(f"Pipe named '{transformer_name}' is not a transformer", exits=1)
        transformer_pipe = cast(Transformer, transformer_pipe_callable)

    piece_encoder = transformer_pipe.model.get_ref("piece_encoder")
    msg.info(f"Found piece encoder: {piece_encoder.name}")

    train_docs = [eg.predicted for eg in train_corpus(nlp)]
    dev_docs = [eg.predicted for eg in dev_corpus(nlp)]

    msg.divider(f"Training corpus statistics")
    print_piece_stats(piece_encoder, train_docs)
    msg.divider(f"Development corpus statistics")
    print_piece_stats(piece_encoder, dev_docs)


def print_piece_stats(piece_encoder: Tok2PiecesModelT, docs: List[Doc]):
    docs_pieces = piece_encoder.predict(docs)

    lens = []
    for doc_pieces in docs_pieces:
        doc_piece_lens = doc_pieces.lengths
        lens.extend(doc_piece_lens.tolist())

    lens_xp = numpy.array(lens)

    msg.text(f"Median token length: {numpy.median(lens_xp)}")
    msg.text(f"Mean token length: {numpy.mean(lens_xp):.2f}")
    msg.text(f"Token length range: [{lens_xp.min()}, {lens_xp.max()}]")
