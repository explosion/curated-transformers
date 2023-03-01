from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from spacy import Errors, Language, Vocab
from spacy.tokens import Doc
from spacy.pipeline import TrainablePipe
from spacy.training import Example, validate_examples, validate_get_examples
from spacy.util import minibatch
from thinc.api import Config, Optimizer, set_dropout_rate
from thinc.model import Model
from thinc.types import Ragged

from ..models.output import DocTransformerOutput, TransformerModelOutput
from ..models.listeners import TransformerListener

DEFAULT_CONFIG_STR = """
    [transformer]

    [transformer.model]
    @architectures = "curated-transformers.XlmrTransformer.v1"

    [transformer.model.piece_encoder]
    @architectures = "curated-transformers.XlmrSentencepieceEncoder.v1"

    [transformer.model.with_spans]
    @architectures = "curated-transformers.WithStridedSpans.v1"
"""


DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)
DOC_EXT_ATTR = "trf_data"


@Language.factory(
    "curated_transformer",
    assigns=["doc._.trf_data"],
    default_config=DEFAULT_CONFIG["transformer"],
)
def make_transformer(
    nlp: Language,
    name: str,
    model: Model,
    *,
    frozen: bool = False,
    all_layer_outputs: bool = False,
) -> "Transformer":
    """Construct a Transformer component, which lets you plug a pre-trained
    transformer model into spaCy so you can use it in your pipeline. One or
    more subsequent spaCy components can use the transformer outputs as features
    in its model, with gradients backpropagated to the single shared weights.

    nlp (Language):
        The pipeline.
    name (str):
        The component instance name.
    model (Model):
        One of the supported pre-trained transformer models.
    frozen (bool):
        If `True`, the model's weights are frozen and no backpropagation is performed.
    all_layer_outputs (bool):
        If `True`, the model returns the outputs of all the layers. Otherwise, only the
        output of the last layer is returned. This must be set to `True` if any of the pipe's
        downstream listeners require the outputs of all transformer layers.
    """
    return Transformer(
        nlp.vocab,
        model,
        name=name,
        frozen=frozen,
        all_layer_outputs=all_layer_outputs,
    )


class Transformer(TrainablePipe):
    """spaCy pipeline component that provides access to a pre-trained transformer
    model from. Downstream components are connected to this pip using TransformerListener
    layers. This works similarly to spaCy's Transformer component and TransformerListener
    sublayer.

    The activations from the transformer are saved in the doc._.trf_data extension
    attribute.
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        *,
        name: str = "transformer",
        frozen: bool = False,
        all_layer_outputs: bool = False,
    ) -> None:
        """
        vocab (Vocab):
            The shared vocabulary.
        model (Model):
            One of the supported pre-trained transformer models.
        name (str):
            The component instance name.
        frozen (bool):
            If `True`, the model's weights are frozen and no backpropagation is performed.
        all_layer_outputs (bool):
            If `True`, the model returns the outputs of all the layers. Otherwise, only the
            output of the last layer is returned. This must be set to `True` if any of the pipe's
            downstream listeners require the outputs of all transformer layers.
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        self.listener_map: Dict[str, List[TransformerListener]] = {}
        self.cfg: Dict[str, Any] = {}

        _install_extensions()
        self.frozen = frozen
        self.all_layer_outputs = all_layer_outputs
        self._set_model_all_layer_outputs(all_layer_outputs)

    @property
    def listeners(self) -> List[TransformerListener]:
        """
        RETURNS (List[TransformerListener]):
            The listener models listening to this component. Usually internals.
        """
        return [m for c in self.listening_components for m in self.listener_map[c]]

    @property
    def listening_components(self) -> List[str]:
        """
        RETURNS (List[str]):
            The downstream components listening to this component. Usually internals.
        """
        return list(self.listener_map.keys())

    def add_listener(self, listener: TransformerListener, component_name: str) -> None:
        """Add a listener for a downstream component. Usually internals."""
        self.listener_map.setdefault(component_name, [])
        if listener not in self.listener_map[component_name]:
            self.listener_map[component_name].append(listener)

    def remove_listener(
        self, listener: TransformerListener, component_name: str
    ) -> bool:
        """Remove a listener for a downstream component. Usually internals.

        RETURNS (bool):
            `True` if successful, `False` otherwise.
        """
        if component_name in self.listener_map:
            if listener in self.listener_map[component_name]:
                self.listener_map[component_name].remove(listener)
                # If no listeners are left, remove entry
                if not self.listener_map[component_name]:
                    del self.listener_map[component_name]
                return True
        return False

    def find_listeners(self, component: Any) -> None:
        """Walk over a model of a processing component, looking for layers that
        are TransformerListener subclasses that have an upstream_name that matches
        this component. Listeners can also set their upstream_name attribute to
        the wildcard string '*' to match any `Transformer`.
        You're unlikely to ever need multiple `Transformer` components, so it's
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

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the pipe to a stream of documents. This usually happens under
        the hood when the nlp object is called on a text and all components are
        applied to the Doc.

        stream (Iterable[Doc]):
            A stream of documents.
        batch_size (int):
            The number of documents to buffer.

        YIELDS (Doc):
            Processed documents in order.
        """
        _install_extensions()
        for batch in minibatch(stream, batch_size):
            preds = self.predict(batch)
            self.set_annotations(batch, preds)
            yield from batch

    def predict(self, docs: Iterable[Doc]) -> TransformerModelOutput:
        """Apply the pipeline's model to a batch of docs, without modifying them.

        docs (Iterable[Doc]):
            The documents to predict.

        RETURNS (TransformerModelOutput):
            The extracted features of each document.
        """
        _install_extensions()
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            width = self.model.get_dim("nO")
            return TransformerModelOutput(
                outputs=[
                    [
                        Ragged(
                            data=self.model.ops.alloc2f(0, width),
                            lengths=self.model.ops.alloc1i(0),
                        )
                    ]
                    for _ in docs
                ],
                last_layer_only=True,
            )

        # To ensure that the model's internal state is always consistent with the pipe's.
        self._set_model_all_layer_outputs(self.all_layer_outputs)
        return self.model.predict(docs)

    def set_annotations(
        self, docs: Sequence[Doc], trf_output: TransformerModelOutput
    ) -> None:
        """Assign the extracted features to the Doc objects. By default, a
        DocTransformerOutput object is written to the doc._.trf_data attribute.

        docs (Iterable[Doc]):
            The documents to modify.
        trf_output (TransformerModelOutput):
            The outputs of the transformer model.
        """
        for doc, tokvecs in zip(docs, trf_output.all_outputs):
            doc._.trf_data = DocTransformerOutput(
                all_outputs=tokvecs, last_layer_only=trf_output.last_layer_only
            )

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Prepare for an update to the transformer.

        Like the `Tok2Vec` component, the `Transformer` component is unusual
        in that it does not receive "gold standard" annotations to calculate
        a weight update. The optimal output of the transformer data is unknown;
        it's a hidden layer inside the network that is updated by backpropagating
        from output layers.

        The `Transformer` component therefore does not perform a weight update
        during its own `update` method. Instead, it runs its transformer model
        and communicates the output and the backpropagation callback to any
        downstream components that have been connected to it via the
        TransformerListener sublayer. If there are multiple listeners, the last
        layer will actually backprop to the transformer and call the optimizer,
        while the others simply increment the gradients.

        examples (Iterable[Example]):
            A batch of Example objects. Only the `predicted` doc object is used,
            the reference doc is ignored.
        drop (float):
            The dropout rate.
        sgd (thinc.api.Optimizer):
            The optimizer.
        losses (Dict[str, float]):
            Optional record of the loss during training. Updated using the component
            name as the key.

        RETURNS (Dict[str, float]):
            The updated losses dictionary.
        """
        if losses is None:
            losses = {}
        validate_examples(examples, "Transformer.update")
        docs = [eg.predicted for eg in examples]
        set_dropout_rate(self.model, drop)
        losses.setdefault(self.name, 0.0)

        # To ensure that the model's internal state is always consistent with the pipe's.
        self._set_model_all_layer_outputs(self.all_layer_outputs)

        outputs, accum_func, backprop_func = self._create_backprops(
            docs, losses, sgd=sgd
        )

        batch_id = TransformerListener.get_batch_id(docs)
        for listener in self.listeners[:-1]:
            listener.receive(batch_id, outputs, accum_func)
        if self.listeners:
            self.listeners[-1].receive(batch_id, outputs, backprop_func)
        return losses

    def get_loss(self, examples: Iterable[Example], scores: Any) -> None:
        """A noop function, for compatibility with the Pipe API. See the `update`
        method for an explanation of the loss mechanics of the component.
        """
        pass

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
        encoder_loader: Optional[Callable] = None,
        piecer_loader: Optional[Callable] = None,
    ):
        """Initialize the pipe for training, using data examples if available.

        get_examples (Callable[[], Iterable[Example]]):
            Optional function that returns gold-standard Example objects.
        nlp (Language):
            The current nlp object.
        encoder_loader (Optional[Callable]):
            Initialization callback for the transformer model.
        piece_loader (Optional[Callable]):
            Initialization callback for the input piece encoder.
        """
        validate_get_examples(get_examples, "Transformer.initialize")

        if encoder_loader:
            self.model.get_ref("transformer").init = encoder_loader  # type: ignore
        if piecer_loader:
            self.model.get_ref("piece_encoder").init = piecer_loader  # type: ignore

        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
        assert doc_sample, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample)

    def add_label(self, label: Any):
        raise NotImplementedError

    def finish_update(self, sgd: Optimizer) -> None:
        """Update parameters using the current parameter gradients.
        The Optimizer instance contains the functionality to perform
        the stochastic gradient descent.

        This method is a noop when the pipe is frozen.

        sgd (thinc.api.Optimizer): The optimizer.

        DOCS: https://spacy.io/api/pipe#finish_update
        """
        if not self.frozen:
            self.model.finish_update(sgd)

    def _create_backprops(
        self,
        docs: Iterable[Doc],
        losses: Dict[str, float],
        *,
        sgd: Optional[Optimizer] = None,
    ) -> Tuple[TransformerModelOutput, Callable, Callable]:
        ops = self.model.ops
        # Accumulate the pipe's loss on the GPU at first.
        cum_loss = ops.xp.full(1, losses[self.name])

        if self.frozen:
            # Ensures that the inner torch model is executed in a `no_grad` context.
            outputs = self.model.predict(docs)
            bp_outputs = None
        else:
            outputs, bp_outputs = self.model.begin_update(docs)

        # We'll use this as a buffer for loss accumulation over individual gradients.
        d_outputs = [
            [Ragged(ops.alloc_f(t2v.dataXd.shape), t2v.lengths) for t2v in doc_layers]
            for doc_layers in outputs.all_outputs
        ]

        # The loss of the transformer is calculated in a different manner than that of the
        # tok2vec component. Instead of updating the loss value with each downstream component's
        # gradient, we accumulate the gradients and and update it just once. This reduces the overhead
        # of launching the associated kernels on the GPU as this operation would otherwise be
        # performed on a per-document, per-layer basis. The resultant loss value, while potentially
        # (slightly) different, still communicates the same information as before w.r.t the training
        # progress: how large were the gradients of the downstream components in this step compared
        # to the previous steps.

        def accumulate_gradient(
            one_d_outputs: List[List[Ragged]], outputs_to_backprop: Tuple[int, ...]
        ) -> None:
            """Accumulate transformer loss and gradient. This is passed as a callback
            to all but the last listener. Only the last one does the backprop.

            `outputs_to_backprop` is a tuple of indices indicating to which layers/outputs
            the gradients are to be propagated.
            """
            nonlocal d_outputs
            for i in range(len(one_d_outputs)):
                for j in outputs_to_backprop:
                    d_outputs[i][j].data += one_d_outputs[i][j].data

        def update_loss():
            """Reduce the gradient buffer and update the losses dict."""
            nonlocal cum_loss
            for i in range(len(d_outputs)):
                for j in range(len(d_outputs[i])):
                    cum_loss += (d_outputs[i][j].data ** 2).sum()
            losses[self.name] = float(cum_loss)

        def backprop(
            one_d_outputs: List[List[Ragged]], outputs_to_backprop: Tuple[int, ...]
        ) -> Any:
            """Callback to actually do the backprop. Passed to last listener."""
            nonlocal d_outputs
            assert bp_outputs is not None
            accumulate_gradient(one_d_outputs, outputs_to_backprop=outputs_to_backprop)
            update_loss()
            d_docs = bp_outputs(d_outputs)
            if sgd is not None:
                self.finish_update(sgd)
            return d_docs

        def backprop_noop(
            one_d_outputs: List[List[Ragged]], outputs_to_backprop: Tuple[int, ...]
        ) -> Any:
            accumulate_gradient(one_d_outputs, outputs_to_backprop=outputs_to_backprop)
            update_loss()
            if sgd is not None:
                self.finish_update(sgd)
            return []

        return outputs, accumulate_gradient, backprop_noop if self.frozen else backprop

    def _set_model_all_layer_outputs(self, new_value: bool):
        self.model.get_ref("transformer").attrs["_all_layer_outputs"] = new_value


def _install_extensions() -> None:
    if not Doc.has_extension(DOC_EXT_ATTR):
        Doc.set_extension(DOC_EXT_ATTR, default=None)
