from typing import Callable, List, Tuple
from spacy.tokens import Doc
from thinc.api import Linear, Model, chain, with_array
from thinc.types import Floats2d


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
