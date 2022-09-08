from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from spacy import Errors, Language, Vocab
from spacy.tokens import Doc
from spacy.pipeline import TrainablePipe
from spacy.training import Example, validate_examples, validate_get_examples
from thinc.api import Optimizer, set_dropout_rate
from thinc.model import Model
from thinc.types import Floats2d, Ragged


DOC_EXT_ATTR = "trf_data"


DEFAULT_TRANSFORMER_MODEL = """
    [model]
    @architectures = "spacy-experimental.XLMRTransformer.v1"
    hf_model_name = "xlm-roberta-base"

    [model.with_spans]
    @architectures = "spacy-experimental.WithStridedSpans.v1"
"""


@Language.factory(
    "experimental_transformer",
    assigns=["doc._.trf_data"],
    default_config={"model": DEFAULT_TRANSFORMER_MODEL},
)
def make_transformer(nlp: Language, name: str, model: Model) -> "Transformer":
    return Transformer(nlp.vocab, model, name)


def last_transformer_layer_listener_v1(
    width: int, pooling: Model[Ragged, Floats2d], upstream: str = "*"
):
    tok2vec = LastTransformerLayerListener(
        upstream_name=upstream, pooling=pooling, width=width
    )
    return tok2vec


class TransformerListener(Model):
    @classmethod
    def get_batch_id(cls, inputs: Iterable[Doc]) -> int:
        """Calculate a content-sensitive hash of the batch of documents, to check
        whether the next batch of documents is unexpected.
        """
        return sum(sum(token.orth for token in doc) for doc in inputs)

    def receive(self, batch_id: int, outputs, backprop) -> None:
        """Store a batch of training predictions and a backprop callback. The
        predictions and callback are produced by the upstream Tok2Vec component,
        and later will be used when the listener's component's model is called.
        """
        self._batch_id = batch_id
        self._outputs = outputs
        self._backprop = backprop

    def verify_inputs(self, inputs) -> bool:
        """Check that the batch of Doc objects matches the ones we have a
        prediction for.
        """
        if self._batch_id is None and self._outputs is None:
            raise ValueError(Errors.E954)
        else:
            batch_id = self.get_batch_id(inputs)
            if batch_id != self._batch_id:
                raise ValueError(Errors.E953.format(id1=batch_id, id2=self._batch_id))
            else:
                return True


class LastTransformerLayerListener(TransformerListener):
    name = "last_transformer_layer_listener"

    def __init__(
        self, upstream_name: str, pooling: Model[Ragged, Floats2d], width: int
    ) -> None:
        """
        upstream_name (str): A string to identify the 'upstream' Tok2Vec component
            to communicate with. The upstream name should either be the wildcard
            string '*', or the name of the `Tok2Vec` component. You'll almost
            never have multiple upstream Tok2Vec components, so the wildcard
            string will almost always be fine.
        width (int):
            The width of the vectors produced by the upstream tok2vec component.
        """
        Model.__init__(
            self, name=self.name, forward=forward, dims={"nO": width}, layers=[pooling]
        )
        self.upstream_name = upstream_name
        self._batch_id: Optional[int] = None
        self._outputs = None
        self._backprop = None


def forward(model: LastTransformerLayerListener, docs, is_train: bool):
    """Supply the outputs from the upstream Tok2Vec component."""
    pooling: Model[Ragged, Floats2d] = model.layers[0]

    outputs = []
    if is_train:
        model.verify_inputs(docs)
        backprops = []
        for output in model._outputs:
            output, pooling_backprop = pooling(output, is_train)
            outputs.append(output)
            backprops.append(pooling_backprop)

        def backprop(dYs):
            dX_pooling = [bp_pool(dY) for dY, bp_pool in zip(dYs, backprops)]
            model._backprop(dX_pooling)
            model._batch_id = None
            model._outputs = None
            model._backprop = None

        return outputs, backprop
    else:
        width = model.get_dim("nO")
        for doc in docs:
            if doc._.trf_data is None:
                outputs.append(model.ops.alloc2f(len(doc), width))
            else:
                output, _ = pooling(doc._.trf_data, is_train)
                outputs.append(output)

        return outputs, lambda dX: []


class Transformer(TrainablePipe):
    """Apply a "token-to-vector" model and set its outputs in the doc._.trf_data
    attribute. This is mostly useful to share a single subnetwork between multiple
    components, e.g. to have one embedding and CNN network shared between a
    parser, tagger and NER.
    In order to use the `Tok2Vec` predictions, subsequent components should use
    the `TransformerListener` layer as the tok2vec subnetwork of their model. This
    layer will read data from the `doc._.trf_data` attribute during prediction.
    During training, the `Tok2Vec` component will save its prediction and backprop
    callback for each batch, so that the subsequent components can backpropagate
    to the shared weights. This implementation is used because it allows us to
    avoid relying on object identity within the models to achieve the parameter
    sharing.
    """

    def __init__(self, vocab: Vocab, model: Model, name: str = "transformer") -> None:
        """Initialize a tok2vec component.
        vocab (Vocab): The shared vocabulary.
        model (thinc.api.Model[List[Doc], List[Floats2d]]):
            The Thinc Model powering the pipeline component. It should take
            a list of Doc objects as input, and output a list of 2d float arrays.
        name (str): The component instance name.
        DOCS: https://spacy.io/api/tok2vec#init
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        self.listener_map: Dict[str, List["TransformerListener"]] = {}
        self.cfg: Dict[str, Any] = {}
        install_extensions()

    @property
    def listeners(self) -> List["TransformerListener"]:
        """RETURNS (List[TransformerListener]): The listener models listening to this
        component. Usually internals.
        """
        return [m for c in self.listening_components for m in self.listener_map[c]]

    @property
    def listening_components(self) -> List[str]:
        """RETURNS (List[str]): The downstream components listening to this
        component. Usually internals.
        """
        return list(self.listener_map.keys())

    def add_listener(
        self, listener: "TransformerListener", component_name: str
    ) -> None:
        """Add a listener for a downstream component. Usually internals."""
        self.listener_map.setdefault(component_name, [])
        if listener not in self.listener_map[component_name]:
            self.listener_map[component_name].append(listener)

    def remove_listener(
        self, listener: "TransformerListener", component_name: str
    ) -> bool:
        """Remove a listener for a downstream component. Usually internals."""
        if component_name in self.listener_map:
            if listener in self.listener_map[component_name]:
                self.listener_map[component_name].remove(listener)
                # If no listeners are left, remove entry
                if not self.listener_map[component_name]:
                    del self.listener_map[component_name]
                return True
        return False

    def find_listeners(self, component) -> None:
        """Walk over a model of a processing component, looking for layers that
        are Tok2vecListener subclasses that have an upstream_name that matches
        this component. Listeners can also set their upstream_name attribute to
        the wildcard string '*' to match any `Tok2Vec`.
        You're unlikely to ever need multiple `Tok2Vec` components, so it's
        fine to leave your listeners upstream_name on '*'.
        """
        names = ("*", self.name)
        if isinstance(getattr(component, "model", None), Model):
            for node in component.model.walk():
                if (
                    isinstance(node, TransformerListener)
                    and node.upstream_name in names
                ):
                    self.add_listener(node, component.name)

    def predict(self, docs: Iterable[Doc]):
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Returns a single tensor for a batch of documents.
        docs (Iterable[Doc]): The documents to predict.
        RETURNS: Vector representations for each token in the documents.
        DOCS: https://spacy.io/api/tok2vec#predict
        """
        install_extensions()
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            width = self.model.get_dim("nO")
            return [
                Ragged(self.model.ops.alloc((0, width), self.model.ops.alloc1i(0)))
                for doc in docs
            ]
        return self.model.predict(docs)

    def set_annotations(self, docs: Sequence[Doc], tokvecses) -> None:
        """Modify a batch of documents, using pre-computed scores.
        docs (Iterable[Doc]): The documents to modify.
        tokvecses: The tensors to set, produced by Tok2Vec.predict.
        DOCS: https://spacy.io/api/tok2vec#set_annotations
        """
        for doc, tokvecs in zip(docs, tokvecses):
            doc._.trf_data = tokvecs

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ):
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model.
        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.
        DOCS: https://spacy.io/api/tok2vec#update
        """
        if losses is None:
            losses = {}
        validate_examples(examples, "Transformer.update")
        docs = [eg.predicted for eg in examples]
        set_dropout_rate(self.model, drop)
        tokvecs, bp_tokvecs = self.model.begin_update(docs)
        d_tokvecs = [
            Ragged(self.model.ops.alloc_f(t2v.dataXd.shape), t2v.lengths)
            for t2v in tokvecs
        ]
        losses.setdefault(self.name, 0.0)

        def accumulate_gradient(one_d_tokvecs):
            """Accumulate tok2vec loss and gradient. This is passed as a callback
            to all but the last listener. Only the last one does the backprop.
            """
            nonlocal d_tokvecs
            for i in range(len(one_d_tokvecs)):
                d_tokvecs[i].data += one_d_tokvecs[i].data
                losses[self.name] += float((one_d_tokvecs[i].data ** 2).sum())
            # return [self.model.ops.alloc2f(*t2v.shape) for t2v in tokvecs]

        def backprop(one_d_tokvecs):
            """Callback to actually do the backprop. Passed to last listener."""
            accumulate_gradient(one_d_tokvecs)
            d_docs = bp_tokvecs(d_tokvecs)
            if sgd is not None:
                self.finish_update(sgd)
            return d_docs

        batch_id = TransformerListener.get_batch_id(docs)
        for listener in self.listeners[:-1]:
            listener.receive(batch_id, tokvecs, accumulate_gradient)
        if self.listeners:
            self.listeners[-1].receive(batch_id, tokvecs, backprop)
        return losses

    def get_loss(self, examples, scores) -> None:
        pass

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.
        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Language): The current nlp object the component is part of.
        DOCS: https://spacy.io/api/tok2vec#initialize
        """
        validate_get_examples(get_examples, "Transformer.initialize")
        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
        assert doc_sample, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample)

    def add_label(self, label):
        raise NotImplementedError


def install_extensions() -> None:
    if not Doc.has_extension(DOC_EXT_ATTR):
        Doc.set_extension(DOC_EXT_ATTR, default=None)
