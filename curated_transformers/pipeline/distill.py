from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from itertools import islice
from spacy import Language, Vocab
from spacy.errors import Errors
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import (
    Example,
    validate_distillation_examples,
    validate_get_examples,
)
from thinc.api import Config, Model, Ops, Optimizer, Ragged, set_dropout_rate

from ..models.pooling import with_ragged_layers
from ..models.distill import MSELoss

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
    nlp: Language, name: str, model: Model, loss_normalization: str = "squared_l2_norm"
) -> "TransformerDistiller":
    """Construct a Transformer distiller component, this component is a no-op
    in regular pipelines, but applies transformer layer losses in distillation.

    vocab (Vocab):
        The shared vocabulary.
    name (str):
        The component instance name.
    model (Model):
        Listerer and mapping (if when necessary)
    loss_normalization (str):
        Type of normalization applied to the MSE loss between the hidden
        layers of the teacher and the student. Supported values:
            "mean" - Takes the mean of all the columns.
            "squared_l2_norm" - The loss is normalized with the L2 norm of the columns and the result is averaged.
    """
    return TransformerDistiller(
        nlp.vocab, model, name=name, loss_normalization=loss_normalization
    )


class TransformerDistiller(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        *,
        name: str = "transformer_distiller",
        loss_normalization: str = "squared_l2_norm",
    ) -> None:
        self.vocab = vocab
        self.model = model
        self.name = name
        self.layer_mapping = None
        self.mse_loss = MSELoss(self.model.ops, normalization=loss_normalization)

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
        examples: Iterable[Example],
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

        validate_distillation_examples(examples, "Distiller.distill")

        student_hidden, student_backprop = self.model.begin_update(
            [eg.predicted for eg in examples]
        )

        # We have to pool the teacher output in the same way as the student.
        student_pooler = self.model.get_ref("tok2vec").get_ref("pooling")

        teacher_hidden = with_ragged_layers(student_pooler).predict(
            [eg.reference for eg in examples]
        )

        layer_mapping = self._layer_mapping(teacher_hidden, student_hidden)

        # Transpose the representations to the get the following shape: [layer, batch, seq, repr]
        student_hidden_t = list(zip(*student_hidden))
        teacher_hidden_t = list(zip(*teacher_hidden))

        # Allocate loss accumulators on the GPU if possible.
        cum_loss = self.model.ops.alloc1f(1)
        d_mses_t = []
        for student_layer_idx, teacher_layer_idx in layer_mapping:
            layer_loss, layer_grads = self.mse_loss(
                student_hidden_t[student_layer_idx], teacher_hidden_t[teacher_layer_idx]
            )
            d_mses_t.append(layer_grads)
            cum_loss += layer_loss

        losses[self.name] += float(cum_loss)

        # Transpose the gradients again to revert back to the original shape: [batch, layer, seq, repr]
        student_backprop(list(zip(*d_mses_t)))
        if sgd is not None:
            self.finish_update(sgd)

        return losses

    def initialize(self, get_examples, *, nlp=None):
        """Initialize the pipe for distillation, using a representative set
        of data examples.

        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects..
        nlp (Language): The current nlp object the component is part of.
        """
        validate_get_examples(get_examples, "Distiller.initialize")
        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample)

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        return iter(stream)
