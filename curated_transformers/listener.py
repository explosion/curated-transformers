from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    cast,
)

from spacy import Errors
from spacy.tokens import Doc
from thinc.model import Model
from thinc.types import Ragged, Floats2d

from .models.output import DocTransformerOutput, TransformerModelOutput


def build_last_transformer_layer_listener_v1(
    width: int,
    pooling: Model[Ragged, Floats2d],
    upstream: str = "*",
    grad_factor: float = 1.0,
) -> Model[List[Doc], List[Floats2d]]:
    tok2vec = LastTransformerLayerListener(
        upstream_name=upstream, pooling=pooling, width=width, grad_factor=grad_factor
    )
    return tok2vec


def build_scalar_weighting_listener_v1(
    width: int,
    weighting: Model[List[Ragged], Ragged],
    pooling: Model[Ragged, Floats2d],
    upstream: str = "*",
    grad_factor: float = 1.0,
) -> Model[List[Doc], List[Floats2d]]:
    tok2vec = ScalarWeightingListener(
        upstream_name=upstream,
        weighting=weighting,
        pooling=pooling,
        width=width,
        grad_factor=grad_factor,
    )
    return tok2vec


class TransformerListener(Model):
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
        predictions and callback are produced by the upstream Tok2Vec component,
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
        self,
        upstream_name: str,
        pooling: Model[Ragged, Floats2d],
        width: int,
        grad_factor: float,
    ) -> None:
        """
        upstream_name (str): A string to identify the 'upstream' Tok2Vec component
            to communicate with. The upstream name should either be the wildcard
            string '*', or the name of the `Tok2Vec` component. You'll almost
            never have multiple upstream Tok2Vec components, so the wildcard
            string will almost always be fine.
        width (int):
            The width of the vectors produced by the upstream tok2vec component.
        grad_factor (float):
            Factor to multiply gradients with.
        """
        Model.__init__(
            self,
            name=self.name,
            forward=last_transformer_layer_listener_forward,
            dims={"nO": width},
            layers=[pooling],
            attrs={"grad_factor": grad_factor},
        )
        self.upstream_name = upstream_name
        self._batch_id = None
        self._outputs = None
        self._backprop = None


def last_transformer_layer_listener_forward(
    model: LastTransformerLayerListener, docs: Iterable[Doc], is_train: bool
) -> Tuple[List[Floats2d], Callable[[Any], Any]]:
    """Supply the outputs from the upstream Tok2Vec component."""
    pooling: Model[Ragged, Floats2d] = model.layers[0]
    grad_factor: float = model.attrs["grad_factor"]

    outputs = []
    if is_train:
        model.verify_inputs(docs)
        backprops = []
        assert model._outputs is not None
        for output in model._outputs.last_hidden_layer_states:
            output, pooling_backprop = pooling(output, is_train)
            outputs.append(output)
            backprops.append(pooling_backprop)

        def backprop(dYs):
            dX_pooling = [[bp_pool(dY)] for bp_pool, dY in zip(backprops, dYs)]
            if grad_factor != 1.0:
                for dx in dX_pooling:
                    dx.data *= grad_factor
            dX = model._backprop(dX_pooling, outputs_to_backprop=(-1,))
            model._batch_id = None
            model._outputs = None
            model._backprop = None
            return dX

        return outputs, backprop
    else:
        width = model.get_dim("nO")
        for doc in docs:
            if doc._.trf_data is None:
                outputs.append(model.ops.alloc2f(len(doc), width))
            else:
                output, _ = pooling(doc._.trf_data.last_hidden_layer_state, is_train)
                outputs.append(output)

        return outputs, lambda dX: []


class ScalarWeightingListener(TransformerListener):
    name = "scalar_weighting_listener"

    def __init__(
        self,
        upstream_name: str,
        weighting: Model[List[Ragged], Ragged],
        pooling: Model[Ragged, Floats2d],
        width: int,
        grad_factor: float,
    ) -> None:
        """
        upstream_name (str): A string to identify the 'upstream' Tok2Vec component
            to communicate with. The upstream name should either be the wildcard
            string '*', or the name of the `Tok2Vec` component. You'll almost
            never have multiple upstream Tok2Vec components, so the wildcard
            string will almost always be fine.`
        width (int):
            The width of the vectors produced by the upstream tok2vec component.
        grad_factor (float):
            Factor to multiply gradients with.
        """
        Model.__init__(
            self,
            name=self.name,
            forward=scalar_weighting_listener_forward,
            dims={"nO": width},
            layers=[weighting, pooling],
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
    """Supply the outputs from the upstream Tok2Vec component."""
    weighting: Model[List[Ragged], Ragged] = model.layers[0]
    pooling: Model[Ragged, Floats2d] = model.layers[1]
    grad_factor: float = model.attrs["grad_factor"]

    invalid_outputs_err_msg = (
        "Scalar layer weighting requires all transformer layer outputs to function - "
        "the upstream transformer's 'all_layer_outputs' property must be set to 'True'"
    )

    outputs = []
    if is_train:
        assert model._outputs is not None
        if model._outputs.last_layer_only:
            raise ValueError(invalid_outputs_err_msg)

        model.verify_inputs(docs)
        weighting_inputs = model._outputs.all_outputs

        backprops = []
        outputs_to_backprop = tuple(i for i in range(0, model._outputs.num_outputs))
        for input in weighting_inputs:
            weighted_output, weighting_backprop = weighting(input, is_train)
            output_pooling, pooling_backprop = pooling(weighted_output, is_train)
            outputs.append(output_pooling)
            backprops.append((weighting_backprop, pooling_backprop))

        def backprop(dYs):
            dX_weighting = []
            for (bp_weighting, bp_pool), dY in zip(backprops, dYs):
                dX_pooling = bp_pool(dY)
                dX_weighting.append(bp_weighting(dX_pooling))

            if grad_factor != 1.0:
                for dx_inner in dX_weighting:
                    for dx in dx_inner:
                        dx.data *= grad_factor

            dX = model._backprop(dX_weighting, outputs_to_backprop=outputs_to_backprop)
            model._batch_id = None
            model._outputs = None
            model._backprop = None
            return dX

        return outputs, backprop
    else:
        width = model.get_dim("nO")
        for doc in docs:
            if doc._.trf_data is None:
                outputs.append(model.ops.alloc2f(len(doc), width))
            else:
                trf_data = cast(DocTransformerOutput, doc._.trf_data)
                if trf_data.last_layer_only:
                    raise ValueError(invalid_outputs_err_msg)

                weighted_output, _ = weighting(trf_data.all_outputs, is_train)
                pooling_output, _ = pooling(weighted_output, is_train)
                outputs.append(pooling_output)

        return outputs, lambda dX: []
