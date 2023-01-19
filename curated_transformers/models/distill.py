from typing import Callable, List, Tuple, Union
from spacy.tokens import Doc
from thinc.api import Linear, Model, chain, with_array, Ops
from thinc.types import Floats1d, Floats2d


def _concat_layers(layer):
    return Model(
        "concat_layers",
        forward=_concat_layers_forward,
        layers=[layer],
        init=_concat_layers_init,
    )


def _concat_layers_forward(model: Model, X, is_train: bool) -> Tuple[List, Callable]:
    layers = len(X[0])

    X_concat_layers = [model.ops.xp.hstack(X_doc) for X_doc in X]
    Y_inner, backprop_inner = model.layers[0](X_concat_layers, is_train)
    Y = [model.ops.xp.hsplit(Y_doc, layers) for Y_doc in Y_inner]

    def backprop(dY):
        dY_concat_layers = [model.ops.xp.hstack(dY_doc) for dY_doc in dY]
        dY_inner = backprop_inner(dY_concat_layers)
        return [model.ops.xp.hsplit(dY_doc, layers) for dY_doc in dY_inner]

    return Y, backprop


def _concat_layers_init(model, X=None, Y=None) -> None:
    layer: Model = model.layers[0]
    layer.initialize(
        X=[model.ops.xp.hstack(X_doc) for X_doc in X] if X is not None else X,
        Y=[model.ops.xp.hstack(Y_doc) for Y_doc in Y] if Y is not None else Y,
    )


def build_layer_distill_model_v1(
    tok2vec: Model[List[Doc], List[Floats2d]], teacher_width: int
) -> Model[List[Doc], List[Floats2d]]:
    student_width = tok2vec.get_dim("nO") if tok2vec.has_dim("nO") else None
    student_layers = tok2vec.attrs["layers"]

    if student_width == teacher_width:
        model = tok2vec
    else:
        model = chain(
            tok2vec,
            _concat_layers(with_array(Linear(teacher_width * student_layers))),
        )

    model.set_ref("tok2vec", tok2vec)

    return model


class MSELoss:
    mean_normalization: str = "mean"
    squared_l2_normalization: str = "squared_l2_norm"

    def __init__(self, ops: Ops, *, normalization: str = mean_normalization):
        self.ops = ops
        self.normalization = normalization

        expected_normalizations = (
            self.mean_normalization,
            self.squared_l2_normalization,
        )
        if normalization not in expected_normalizations:
            raise ValueError(
                f"Normalization for MSE loss must be one of the following values: {expected_normalizations}"
            )

    def __call__(
        self, predicted: List[Floats2d], target: List[Floats2d]
    ) -> Tuple[Floats1d, List[Floats2d]]:
        if len(predicted) != len(target):
            raise ValueError(
                f"MSE loss requires inputs with the same batch size, but got: {len(predicted)},{len(target)}"
            )
        batch_size = len(predicted)
        n_elements = predicted[0].size
        cum_loss = self.ops.alloc1f(1)
        grads = []

        for y_h, y in zip(predicted, target):
            if len(y_h.shape) != len(y.shape) or len(y_h.shape) != 2:
                raise ValueError(
                    f"MSE loss requires 2D inputs of the same shape, but got: {y_h.shape},{y.shape}"
                )
            grad = y_h - y
            loss = grad**2

            if self.normalization == self.mean_normalization:
                grad /= batch_size * n_elements
                cum_loss += loss.sum()  # type: ignore
            elif self.normalization == self.squared_l2_normalization:
                norm = self.ops.xp.linalg.norm(y.reshape(-1)) ** 2  # type: ignore
                grad /= norm
                loss /= norm
                cum_loss += loss.sum()  # type: ignore
            grads.append(grad)

        if self.normalization == self.mean_normalization:
            cum_loss /= batch_size * n_elements
        elif self.normalization == self.squared_l2_normalization:
            cum_loss /= batch_size

        return (cum_loss, grads)
