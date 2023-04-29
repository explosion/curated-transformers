from typing import List, Optional, Tuple, Iterator, Any, cast
from dataclasses import dataclass
from collections import Counter

from spacy.tokens import Doc
from thinc.api import Model, Ragged

from .types import (
    WsTokenAdapterInT,
    WsTokenAdapterOutT,
    WsTokenAdapterBackpropT,
    WsTokenAdapterModelT,
)
from ..tokenization.types import Tok2PiecesInT


@dataclass
class TokenAlignment:
    # The piece offset of the token within the doc with whitespace tokens.
    ws_piece_offset: int

    # The piece offset of the token within the doc without whitespace tokens.
    # Whitespace tokens have None as their offset, since they are not
    # represented in piece sequences without whitespace.
    no_ws_piece_offset: Optional[int]

    # The length of this sequence in pieces.
    n_pieces: int

    @property
    def is_whitespace(self) -> bool:
        return self.no_ws_piece_offset is None


@dataclass
class Alignment:
    # Token alignments.
    tokens: List[TokenAlignment]

    # The length of the document in pieces with whitespace tokens.
    ws_n_pieces: int

    # The length of the document in pieces without whitespace tokens.
    no_ws_n_pieces: int

    @property
    def has_no_whitespace(self) -> bool:
        return self.ws_n_pieces == self.no_ws_n_pieces

    def __iter__(self) -> Iterator[TokenAlignment]:
        return iter(self.tokens)


def with_non_ws_tokens(
    layer: Model[Tok2PiecesInT, WsTokenAdapterOutT]
) -> WsTokenAdapterModelT:
    """Removes non-whitespace tokens from the input before
    passing it to the inner model."""
    return Model(
        "with_non_ws_tokens",
        with_non_ws_tokens_forward,
        init=with_non_ws_tokens_init,
        layers=[layer],
    )


def with_non_ws_tokens_forward(
    model: Model, X: WsTokenAdapterInT, is_train: bool
) -> Tuple[WsTokenAdapterOutT, WsTokenAdapterBackpropT]:
    inner: Model[Tok2PiecesInT, WsTokenAdapterOutT] = model.layers[0]
    tokens, ws_counts = _filter_tokens(X)
    Y_no_ws, backprop_no_ws = inner(tokens, is_train)

    # Note: we modify the model outputs in-place. Since we are wrapping the
    # model, there should be no other consumers. Not sure yet if the same
    # applies to the gradient (e.g. consider summing two encoders of the
    # same width downstream.)

    alignments = _create_alignments(model, Y_no_ws, ws_counts)
    _add_whitespace_tokens(model, Y_no_ws, alignments)

    def backprop(dY: List[List[Ragged]]) -> Any:
        _remove_whitespace_tokens(model, dY, alignments)
        backprop_no_ws(dY)
        return []

    return Y_no_ws, backprop


def with_non_ws_tokens_init(
    model: WsTokenAdapterModelT,
    X: Optional[WsTokenAdapterInT] = None,
    Y: Optional[WsTokenAdapterInT] = None,
) -> None:
    model.layers[0].initialize(X=_filter_tokens(X)[0] if X is not None else None, Y=Y)


def _create_alignments(
    model: Model, output: WsTokenAdapterOutT, ws_counts: List[Counter]
) -> List[Alignment]:
    """Create an alignment between whitespace and non-whitespace sequences."""
    alignments = []
    for doc_output, doc_ws_counts in zip(output.all_outputs, ws_counts):
        doc_alignments = []
        no_ws_offset = 0
        ws_offset = 0
        pieces_lens = model.ops.to_numpy(doc_output[0].lengths).tolist()
        for idx, pieces_len in enumerate(pieces_lens):
            # Whitespace tokens that preceded a token.
            n_ws = doc_ws_counts[idx]
            for i in range(ws_offset, ws_offset + n_ws):
                doc_alignments.append(
                    TokenAlignment(
                        ws_piece_offset=i, no_ws_piece_offset=None, n_pieces=1
                    )
                )
            ws_offset += n_ws

            # The token itself.
            doc_alignments.append(
                TokenAlignment(
                    n_pieces=pieces_len,
                    no_ws_piece_offset=no_ws_offset,
                    ws_piece_offset=ws_offset,
                )
            )

            no_ws_offset += pieces_len
            ws_offset += pieces_len

        # There can be spaces after the last non-whitespace token.
        n_ws = doc_ws_counts[len(pieces_lens)]
        for i in range(ws_offset, ws_offset + n_ws):
            doc_alignments.append(
                TokenAlignment(ws_piece_offset=i, no_ws_piece_offset=None, n_pieces=1)
            )
        ws_offset += n_ws

        # Add the doc alignment.
        alignments.append(
            Alignment(
                tokens=doc_alignments,
                no_ws_n_pieces=no_ws_offset,
                ws_n_pieces=ws_offset,
            )
        )

    return alignments


def _filter_tokens(docs: List[Doc]) -> Tuple[Tok2PiecesInT, List[Counter]]:
    """Filter out whitespace tokens. Returns the non-whitespace tokens
    and a mapping from the (non-whitespace) token offset to the number
    of whitespaces that preceded the token."""
    tokens = []
    ws_counts = []
    for doc in docs:
        doc_tokens = []
        doc_ws_counts: Counter = Counter()
        offset = 0
        for token in doc:
            if token.is_space:
                doc_ws_counts[offset] += 1
                continue
            doc_tokens.append(token)
            offset += 1
        tokens.append(doc_tokens)
        ws_counts.append(doc_ws_counts)

    return tokens, ws_counts


def _add_whitespace_tokens(
    model: Model, Y_no_ws: WsTokenAdapterOutT, alignments: List[Alignment]
):
    """Add stub representations for whitespace tokens."""
    for Y_doc, doc_alignment in zip(Y_no_ws.all_outputs, alignments):
        if doc_alignment.has_no_whitespace:
            continue

        hidden_width = Y_doc[0].dataXd.shape[1]
        for layer_idx, layer in enumerate(Y_doc):
            lengths = []
            new_layer = model.ops.alloc2f(doc_alignment.ws_n_pieces, hidden_width)

            for alignment in doc_alignment:
                if not alignment.is_whitespace:
                    assert alignment.no_ws_piece_offset is not None
                    new_layer[
                        alignment.ws_piece_offset : alignment.ws_piece_offset
                        + alignment.n_pieces,
                        :,
                    ] = layer.dataXd[
                        alignment.no_ws_piece_offset : alignment.no_ws_piece_offset
                        + alignment.n_pieces,
                        :,
                    ]
                lengths.append(alignment.n_pieces)

            Y_doc[layer_idx] = Ragged(new_layer, lengths=model.ops.asarray1i(lengths))


def _remove_whitespace_tokens(
    model: Model, dY: List[List[Ragged]], alignments: List[Alignment]
):
    """Remove representations for whitespace tokens."""
    for dY_doc, doc_alignment in zip(dY, alignments):
        if doc_alignment.has_no_whitespace:
            continue

        hidden_width = cast(Tuple[int, ...], dY_doc[0].dataXd.shape)[1]
        for layer_idx, layer in enumerate(dY_doc):
            new_layer = model.ops.alloc2f(doc_alignment.no_ws_n_pieces, hidden_width)
            lengths = []

            for alignment in doc_alignment:
                if alignment.is_whitespace:
                    continue

                assert alignment.no_ws_piece_offset is not None
                # Extra type ignore to accomodate two MyPy versions :(.
                new_layer[  # type: ignore
                    alignment.no_ws_piece_offset : alignment.no_ws_piece_offset
                    + alignment.n_pieces,  # type: ignore
                    :,
                ] = layer.dataXd[  # type: ignore
                    alignment.ws_piece_offset : alignment.ws_piece_offset
                    + alignment.n_pieces,
                    :,
                ]

                lengths.append(alignment.n_pieces)

            dY_doc[layer_idx] = Ragged(new_layer, lengths=model.ops.asarray1i(lengths))
