from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from itertools import islice
from spacy import Language, Vocab
from spacy.errors import Errors
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import validate_get_examples
from spacy.training.example import Example
from thinc.api import Config, Model, Ops, Optimizer, Ragged, set_dropout_rate

from ..models.pooling import with_ragged_layers

DEFAULT_CONFIG_STR = """
    [transformer_distiller]

    [transformer_distiller.model]
    @architectures = "curated-transformers.LayerDistill.v1"
    teacher_width = 768

    [transformer_distiller.model.tok2vec]
    @architectures = "curated-transformers.LastTransformerLayerListener.v1"
    width = 768
    pooling = {"@layers":"reduce_mean.v1"}
    upstream = "*"
"""


DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)


@Language.factory(
    "transformer_distiller",
    assigns=[],
    default_config=DEFAULT_CONFIG["transformer_distiller"],
)
def make_transformer_distiller(
    nlp: Language,
    name: str,
    model: Model,
) -> "TransformerDistiller":
    """Construct a Transformer distiller component, this component is a no-op
    in regular pipelines, but applies transformer layer losses in distillation.

    vocab (Vocab):
        The shared vocabulary.
    name (str):
        The component instance name.
    model (Model):
        Listerer and mapping (if when necessary)
    """
    return TransformerDistiller(
        nlp.vocab,
        model,
        name=name,
    )


class TransformerDistiller(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        *,
        name: str = "transformer_distiller",
    ) -> None:
        self.vocab = vocab
        self.model = model
        self.name = name
        self.layer_mapping = None

    def _layer_mapping(self, teacher_hidden, student_hidden):
        if self.layer_mapping is not None:
            return self.layer_mapping

        t_doc = teacher_hidden[0]
        s_doc = student_hidden[0]

        n_teacher = len(t_doc) - 1
        n_student = len(s_doc) - 1
        if n_student == 0 or n_teacher == 0:
            # Distill the last layer if the teacher or student only has one layer.
            self.layer_mapping = [(-1, -1)]
        elif n_teacher % n_student != 0:
            raise ValueError(
                "Cannot distribute student layers evenly over teacher layers"
            )
        else:
            layer_multiple = n_teacher // n_student
            # Always map the embedding layer.
            self.layer_mapping = [(0, 0)]
            # Uniformly distribute other layers.
            self.layer_mapping.extend(
                (i + 1, (i + 1) * layer_multiple) for i in range(n_student)
            )

        return self.layer_mapping

    def distill(
        self,
        teacher_pipe: Optional["TrainablePipe"],
        teacher_docs: Iterable[Doc],
        student_docs: Iterable[Doc],
        *,
        drop: float = 0.0,
        sgd: Optimizer = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        # Does not have a corresponding pipe in the teacher.
        assert teacher_pipe is None

        if losses is None:
            losses = {}
        set_dropout_rate(self.model, drop)
        losses.setdefault(self.name, 0.0)

        if not any(len(doc) for doc in teacher_docs):
            return losses
        if not any(len(doc) for doc in student_docs):
            return losses

        student_hidden, student_backprop = self.model.begin_update(student_docs)

        # We have to pool the teacher output in the same way as the student.
        student_pooler = self.model.get_ref("tok2vec").get_ref("pooling")

        teacher_hidden = with_ragged_layers(student_pooler).predict(teacher_docs)

        layer_mapping = self._layer_mapping(teacher_hidden, student_hidden)

        normalize = True
        d_mse = []
        for t_doc, s_doc in zip(teacher_hidden, student_hidden):
            if normalize:
                d_mse_doc = [
                    (s_doc[student_layer] - t_doc[teacher_layer])
                    / self.model.ops.xp.linalg.norm(t_doc[teacher_layer])
                    for student_layer, teacher_layer in layer_mapping
                ]
            else:
                d_mse_doc = [
                    s_doc[student_layer] - t_doc[teacher_layer]
                    for student_layer, teacher_layer in layer_mapping
                ]

            d_mse.append(d_mse_doc)

        losses[self.name] += sum(
            sum(float((d_layer**2).sum()) for d_layer in d) for d in d_mse
        )

        student_backprop(d_mse)
        if sgd not in (None, False):
            self.finish_update(sgd)

        return losses

    def initialize(self, get_examples, *, nlp=None):
        """Initialize the pipe for distillation, using a representative set
        of data examples.

        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects..
        nlp (Language): The current nlp object the component is part of.
        """
        validate_get_examples(get_examples, "TransformerDistiller.initialize")
        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample)

    def predict(self, docs: Iterable[Doc]) -> Any:
        return None

    def set_annotations(self, docs: Iterable[Doc], scores) -> None:
        pass

    def update(
        self,
        examples: Iterable["Example"],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        return losses

    def finish_update(self, sgd: Optimizer) -> None:
        pass

    def add_label(self, label: str) -> int:
        return 0

    @property
    def is_trainable(self) -> bool:
        return False
