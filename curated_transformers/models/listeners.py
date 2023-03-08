from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
)

from spacy import Errors as SpacyErrors
from spacy.tokens import Doc
from thinc.model import Model
from thinc.types import Ragged, Floats2d

from .output import TransformerModelOutput
from .pooling import with_ragged_layers, with_ragged_last_layer
from .types import (
    WithRaggedLayersModelT,
    WithRaggedLastLayerModelT,
    PoolingModelT,
    ScalarWeightOutT,
    ScalarWeightModelT,
)
from .output import TransformerModelOutput
from .pooling import with_ragged_layers, with_ragged_last_layer
from .types import (
    WithRaggedLayersModelT,
    WithRaggedLastLayerModelT,
    PoolingModelT,
)
from ..errors import Errors


def build_transformer_layers_listener_v1(
    layers: int,
    width: int,
    pooling: PoolingModelT,
    upstream: str = "*",
    grad_factor: float = 1.0,
) -> Model[List[Doc], List[Floats2d]]:
    """Construct a listener layer that communicates with one or more upstream Transformer
    components. This layer extracts the output of all transformer layers and performs
    pooling over the individual pieces of each Doc token, returning their corresponding
    representations.

    layers (int):
        The the number of layers produced by the upstream transformer component,
        excluding the embedding layer.
    width (int):
        The width of the vectors produced by the upstream transformer component.
    pooling (PoolingModelT):
        Model that is used to perform pooling over the piece representations.
    upstream_name (str):
        A string to identify the 'upstream' Transformer component
        to communicate with. The upstream name should either be the wildcard
        string '*', or the name of the Transformer component.

        In almost all cases, the wildcard string will suffice as there'll only be one
        upstream Transformer component. But in certain situations, e.g: you have disjoint
        datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
        downstream task requires its own token representations, you could end up with
        more than one Transformer component in the pipeline.
    grad_factor (float):
        Factor to multiply gradients with.
    """
    transformer = TransformerLayersListener(
        upstream_name=upstream,
        pooling=pooling,
        layers=layers,
        width=width,
        grad_factor=grad_factor,
    )
    return transformer


def build_last_transformer_layer_listener_v1(
    width: int,
    pooling: PoolingModelT,
    upstream: str = "*",
    grad_factor: float = 1.0,
) -> Model[List[Doc], List[Floats2d]]:
    """Construct a listener layer that communicates with one or more upstream Transformer
    components. This layer extracts the output of the last transformer layer and performs
    pooling over the individual pieces of each Doc token, returning their corresponding
    representations.

    width (int):
        The width of the vectors produced by the upstream transformer component.
    pooling (Model):
        Model that is used to perform pooling over the piece representations.
    upstream_name (str):
        A string to identify the 'upstream' Transformer component
        to communicate with. The upstream name should either be the wildcard
        string '*', or the name of the Transformer component.

        In almost all cases, the wildcard string will suffice as there'll only be one
        upstream Transformer component. But in certain situations, e.g: you have disjoint
        datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
        downstream task requires its own token representations, you could end up with
        more than one Transformer component in the pipeline.
    grad_factor (float):
        Factor to multiply gradients with.
    """
    transformer = LastTransformerLayerListener(
        upstream_name=upstream, pooling=pooling, width=width, grad_factor=grad_factor
    )
    return transformer


def build_scalar_weighting_listener_v1(
    width: int,
    weighting: ScalarWeightModelT,
    pooling: PoolingModelT,
    upstream: str = "*",
    grad_factor: float = 1.0,
) -> Model[List[Doc], List[Floats2d]]:
    """Construct a listener layer that communicates with one or more upstream Transformer
    components. This layer calculates a weighted representation of all transformer layer
    outputs and performs pooling over the individual pieces of each Doc token, returning
    their corresponding representations.

    Requires its upstream Transformer components to return all layer outputs from
    their models.

    width (int):
        The width of the vectors produced by the upstream transformer component.
    weighting (Model):
        Model that is used to perform the weighting of the different layer outputs.
    pooling (Model):
        Model that is used to perform pooling over the piece representations.
    upstream_name (str):
        A string to identify the 'upstream' Transformer component
        to communicate with. The upstream name should either be the wildcard
        string '*', or the name of the Transformer component.

        In almost all cases, the wildcard string will suffice as there'll only be one
        upstream Transformer component. But in certain situations, e.g: you have disjoint
        datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
        downstream task requires its own token representations, you could end up with
        more than one Transformer component in the pipeline.
    grad_factor (float):
        Factor to multiply gradients with.
    """
    transformer = ScalarWeightingListener(
        upstream_name=upstream,
        weighting=weighting,
        pooling=pooling,
        width=width,
        grad_factor=grad_factor,
    )
    return transformer


class TransformerListener(Model):
    """A layer that gets fed its answers from an upstream Transformer component.

    The TransformerListener layer is used as a sublayer within a component such
    as a parser, NER or text categorizer. Usually you'll have multiple listeners
    connecting to a single upstream Transformer component that's earlier in the
    pipeline. These layers act as proxies, passing the predictions
    from the Transformer component into downstream components and communicating
    gradients back upstream.
    """

    upstream_name: str
    _batch_id: Optional[int]
    _outputs: Optional[TransformerModelOutput]
    _backprop: Optional[Callable[[List[List[Ragged]], Tuple[int]], Any]]

    @classmethod
    def get_batch_id(cls, inputs: Iterable[Doc]) -> int:
        """Calculate a content-sensitive hash of the batch of documents, to check
        whether the next batch of documents is unexpected.
        """
        return sum(sum(token.orth for token in doc) for doc in inputs)

    def receive(
        self,
        batch_id: int,
        outputs: TransformerModelOutput,
        backprop: Callable[[List[List[Ragged]], Tuple[int]], Any],
    ) -> None:
        """Store a batch of training predictions and a backprop callback. The
        predictions and callback are produced by the upstream Transformer component,
        and later will be used when the listener's component's model is called.
        """
        self._batch_id = batch_id
        self._outputs = outputs
        self._backprop = backprop

    def verify_inputs(self, inputs: Iterable[Doc]) -> bool:
        """Check that the batch of Doc objects matches the ones we have a
        prediction for.
        """
        if self._batch_id is None and self._outputs is None:
            raise ValueError(SpacyErrors.E954)
        else:
            batch_id = self.get_batch_id(inputs)
            if batch_id != self._batch_id:
                raise ValueError(
                    SpacyErrors.E953.format(id1=batch_id, id2=self._batch_id)
                )
            else:
                return True


class TransformerLayersListener(TransformerListener):
    """Passes through the pooled representations of all transformer layers.

    Requires its upstream Transformer component to return all layer outputs from
    its model.
    """

    name = "transformer_layers_listener"

    def __init__(
        self,
        upstream_name: str,
        pooling: PoolingModelT,
        width: int,
        layers: int,
        grad_factor: float,
    ) -> None:
        """
        upstream_name (str):
            A string to identify the 'upstream' Transformer component
            to communicate with. The upstream name should either be the wildcard
            string '*', or the name of the Transformer component.

            In almost all cases, the wildcard string will suffice as there'll only be one
            upstream Transformer component. But in certain situations, e.g: you have disjoint
            datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
            downstream task requires its own token representations, you could end up with
            more than one Transformer component in the pipeline.
        pooling (PoolingModelT):
            Model that is used to perform pooling over the piece representations.
        width (int):
            The width of the vectors produced by the upstream transformer component.
        layers (int):
            The the number of layers produced by the upstream transformer component,
            excluding the embedding layer.
        grad_factor (float):
            Factor to multiply gradients with.
        """
        Model.__init__(
            self,
            name=self.name,
            forward=tranformer_layers_listener_forward,
            dims={"nO": width},
            layers=[with_ragged_layers(pooling)],
            attrs={
                "grad_factor": grad_factor,
                "layers": layers + 1,
            },
            refs={"pooling": pooling},
        )
        self.upstream_name = upstream_name
        self._batch_id = None
        self._outputs = None
        self._backprop = None


def tranformer_layers_listener_forward(
    model: TransformerLayersListener, docs: Iterable[Doc], is_train: bool
) -> Tuple[List[List[Floats2d]], Callable[[Any], Any]]:
    pooling: WithRaggedLayersModelT = model.layers[0]
    grad_factor: float = model.attrs["grad_factor"]
    n_layers: int = model.attrs["layers"]

    if is_train:
        assert model._outputs is not None
        if model._outputs.last_layer_only:
            raise ValueError(
                Errors.E012.format(listener_name="TransformerLayersListener")
            )

        model.verify_inputs(docs)

        Y, backprop_pooling = pooling(model._outputs.all_outputs, is_train)

        def backprop(dY):
            dX = backprop_pooling(dY)

            if grad_factor != 1.0:
                for dX_doc in dX:
                    for dX_layer in dX_doc:
                        dX_layer.data *= grad_factor

            outputs_to_backprop = tuple(i for i in range(0, model._outputs.num_outputs))
            dX = model._backprop(dX, outputs_to_backprop=outputs_to_backprop)

            model._batch_id = None
            model._outputs = None
            model._backprop = None

            return dX

        return Y, backprop
    else:
        width = model.get_dim("nO")

        no_trf_data = [doc._.trf_data is None for doc in docs]
        if any(no_trf_data):
            assert all(no_trf_data)
            return [
                [model.ops.alloc2f(len(doc), width) for _ in range(n_layers)]
                for doc in docs
            ], lambda dY: []

        if any(doc._.trf_data.last_layer_only for doc in docs):
            raise ValueError(
                Errors.E012.format(listener_name="TransformerLayersListener")
            )

        return pooling.predict(docs), lambda dY: []


class LastTransformerLayerListener(TransformerListener):
    """Extracts the output of the last transformer layer and performs pooling over the
    individual pieces of each Doc token, returning their corresponding representations.
    """

    name = "last_transformer_layer_listener"

    def __init__(
        self,
        upstream_name: str,
        pooling: PoolingModelT,
        width: int,
        grad_factor: float,
    ) -> None:
        """
        upstream_name (str):
            A string to identify the 'upstream' Transformer component
            to communicate with. The upstream name should either be the wildcard
            string '*', or the name of the Transformer component.

            In almost all cases, the wildcard string will suffice as there'll only be one
            upstream Transformer component. But in certain situations, e.g: you have disjoint
            datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
            downstream task requires its own token representations, you could end up with
            more than one Transformer component in the pipeline.
        width (int):
            The width of the vectors produced by the upstream transformer component.
        pooling (Model):
            Model that is used to perform pooling over the piece representations.
        grad_factor (float):
            Factor to multiply gradients with.
        """
        Model.__init__(
            self,
            name=self.name,
            forward=last_transformer_layer_listener_forward,
            dims={"nO": width},
            layers=[with_ragged_last_layer(pooling)],
            attrs={"grad_factor": grad_factor},
            refs={"pooling": pooling},
        )
        self.upstream_name = upstream_name
        self._batch_id = None
        self._outputs = None
        self._backprop = None


def last_transformer_layer_listener_forward(
    model: LastTransformerLayerListener, docs: Iterable[Doc], is_train: bool
) -> Tuple[List[Floats2d], Callable[[Any], Any]]:
    pooling: WithRaggedLastLayerModelT = model.layers[0]
    grad_factor: float = model.attrs["grad_factor"]

    if is_train:
        model.verify_inputs(docs)
        assert model._outputs is not None
        Y, backprop_pooling = pooling(model._outputs.last_hidden_layer_states, is_train)

        def backprop(dY):
            dX_pooling = backprop_pooling(dY)
            if grad_factor != 1.0:
                for dx in dX_pooling:
                    dx.data *= grad_factor
            dX = model._backprop([[d] for d in dX_pooling], outputs_to_backprop=(-1,))
            model._batch_id = None
            model._outputs = None
            model._backprop = None
            return dX

        return Y, backprop
    else:
        width = model.get_dim("nO")

        no_trf_data = [doc._.trf_data is None for doc in docs]
        if any(no_trf_data):
            assert all(no_trf_data)
            return [model.ops.alloc2f(len(doc), width) for doc in docs], lambda dY: []

        return pooling.predict(docs), lambda dY: []


class ScalarWeightingListener(TransformerListener):
    """Calculates a weighted representation of all transformer layer outputs and
    performs pooling over the individual pieces of each Doc token, returning their
    corresponding representations.

    Requires its upstream Transformer component to return all layer outputs from
    its model.
    """

    name = "scalar_weighting_listener"

    def __init__(
        self,
        upstream_name: str,
        weighting: ScalarWeightModelT,
        pooling: PoolingModelT,
        width: int,
        grad_factor: float,
    ) -> None:
        """
        upstream_name (str):
            A string to identify the 'upstream' Transformer component
            to communicate with. The upstream name should either be the wildcard
            string '*', or the name of the Transformer component.

            In almost all cases, the wildcard string will suffice as there'll only be one
            upstream Transformer component. But in certain situations, e.g: you have disjoint
            datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
            downstream task requires its own token representations, you could end up with
            more than one Transformer component in the pipeline.
        weighting (Model):
            Model that is used to perform the weighting of the different layer outputs.
        pooling (Model):
            Model that is used to perform pooling over the piece representations.
        width (int):
            The width of the vectors produced by the upstream transformer component.
        grad_factor (float):
            Factor to multiply gradients with.
        """
        Model.__init__(
            self,
            name=self.name,
            forward=scalar_weighting_listener_forward,
            dims={"nO": width},
            layers=[weighting, with_ragged_last_layer(pooling)],
            attrs={
                "grad_factor": grad_factor,
            },
        )
        self.upstream_name = upstream_name
        self._batch_id = None
        self._outputs = None
        self._backprop = None


def scalar_weighting_listener_forward(
    model: ScalarWeightingListener, docs: Iterable[Doc], is_train: bool
) -> Tuple[List[Floats2d], Callable[[Any], Any]]:
    weighting: ScalarWeightModelT = model.layers[0]
    pooling: WithRaggedLastLayerModelT = model.layers[1]
    grad_factor: float = model.attrs["grad_factor"]

    if is_train:
        assert model._outputs is not None
        if model._outputs.last_layer_only:
            raise ValueError(
                Errors.E012.format(listener_name="ScalarWeightingListener")
            )

        model.verify_inputs(docs)

        Y_weighting: ScalarWeightOutT = []
        weighting_inputs = model._outputs.all_outputs
        outputs_to_backprop = tuple(i for i in range(model._outputs.num_outputs))

        Y_weighting, backprop_weighting = weighting(weighting_inputs, is_train)
        Y, backprop_pooling = pooling(Y_weighting, is_train)

        def backprop(dYs):
            dX_pooling = backprop_pooling(dYs)
            dX_weighting = backprop_weighting(dX_pooling)

            if grad_factor != 1.0:
                for dx_inner in dX_weighting:
                    for dx in dx_inner:
                        dx.data *= grad_factor

            dX = model._backprop(dX_weighting, outputs_to_backprop=outputs_to_backprop)
            model._batch_id = None
            model._outputs = None
            model._backprop = None
            return dX

        return Y, backprop
    else:
        width = model.get_dim("nO")

        no_trf_data = [doc._.trf_data is None for doc in docs]
        if any(no_trf_data):
            assert all(no_trf_data)
            return [model.ops.alloc2f(len(doc), width) for doc in docs], lambda dY: []

        if any(doc._.trf_data.last_layer_only for doc in docs):
            raise ValueError(
                Errors.E012.format(listener_name="ScalarWeightingListener")
            )

        Y_weigthing = weighting.predict([doc._.trf_data.all_outputs for doc in docs])
        Y = pooling.predict(Y_weigthing)

        return Y, lambda dX: []
