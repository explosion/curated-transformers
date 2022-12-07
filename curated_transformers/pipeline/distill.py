from typing import Dict, Iterable, Iterator, Optional
from itertools import islice
from spacy import Language, Vocab
from spacy.errors import Errors
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import validate_get_examples
from thinc.api import Config, Model, Optimizer, set_dropout_rate

DEFAULT_CONFIG_STR = """
    [transformer_distiller]

    [transformer_distiller.model]
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

        student_hidden, student_backprop = self.model.begin_update(student_docs)

        # We have to pool the teacher output in the same way as the student.
        student_pooler = self.model.get_ref("pooling")
        teacher_hidden = [
            student_pooler.predict(doc._.trf_data.last_hidden_layer_state)
            for doc in teacher_docs
        ]

        diffs = [(t - s) for t, s in zip(teacher_hidden, student_hidden)]
        d_mse = [2 * diff for diff in diffs]
        losses[self.name] += sum((diff**2).sum() for diff in diffs)
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

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        return stream
