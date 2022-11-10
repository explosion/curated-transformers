from typing import Any, Callable, List, TypeVar, Union

from spacy.tokens.doc import Doc
from thinc.model import Model
from thinc.types import Floats2d, Ints1d, Ragged

from .output import TransformerModelOutput


WsTokenAdapterInT = List[Doc]
WsTokenAdapterOutT = TransformerModelOutput
WsTokenAdapterBackpropT = Callable[[List[List[Ragged]]], Any]
WsTokenAdapterModelT = Model[WsTokenAdapterInT, WsTokenAdapterOutT]

# In case of a single list, each element corresponds to a single document/span.
# For nested lists, each inner list corresponds to a single document/span.
RaggedInOutT = Union[List[Ragged], List[List[Ragged]]]
Floats2dInOutT = Union[List[Floats2d], List[List[Floats2d]]]

SpanExtractorInT = List[Ragged]
SpanExtractorOutT = TransformerModelOutput
SpanExtractorBackpropT = Callable[[RaggedInOutT], RaggedInOutT]
SpanExtractorModelT = Model[SpanExtractorInT, SpanExtractorOutT]

# Inner PyTorch transformer.
TorchTransformerInT = List[Ints1d]
TorchTransformerOutT = TransformerModelOutput
TorchTransformerModelT = Model[TorchTransformerInT, TorchTransformerOutT]

SentMarkerRemoverInOutT = TransformerModelOutput
SentMarkerRemoverBackpropT = Callable[[RaggedInOutT], RaggedInOutT]
SentMarkerRemoverModelT = Model[SentMarkerRemoverInOutT, SentMarkerRemoverInOutT]

# Wrapper that includes the various pre- and post-processing layers such
# as the piece encoder, adapter, etc.
TransformerInT = List[Doc]
TransformerOutT = TransformerModelOutput
TransformerBackpropT = Callable[[List[List[Floats2d]]], Any]
TransformerModelT = Model[TransformerInT, TransformerOutT]

ScalarWeightInT = List[Ragged]
ScalarWeightOutT = Ragged
ScalarWeightModelT = Model[ScalarWeightInT, ScalarWeightOutT]
