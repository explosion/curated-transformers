from typing import List
from spacy.tokens import Doc
from thinc.api import Linear, Model, chain, with_array, with_flatten
from thinc.types import Floats2d


def build_layer_distill_model_v1(
    tok2vec: Model[List[Doc], List[Floats2d]], teacher_width: int
) -> Model[List[Doc], List[Floats2d]]:
    student_width = tok2vec.get_dim("nO") if tok2vec.has_dim("nO") else None

    if student_width == teacher_width:
        model = tok2vec
    else:
        model = chain(
            tok2vec, with_flatten(with_array(Linear(teacher_width, student_width)))
        )

    model.set_ref("tok2vec", tok2vec)

    return model
