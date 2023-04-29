from typing import List
from thinc.api import Model, chain
from thinc.types import Floats2d, Ragged

from curated_transformers.models.output import TransformerModelOutput
from curated_transformers.models.remove_eos_bos import remove_bos_eos
from curated_transformers.models.with_non_ws_tokens import with_non_ws_tokens


def _mock_transformer() -> Model[List[Floats2d], TransformerModelOutput]:
    def forward(model: Model, X: List[Floats2d], is_train: bool):
        def backprop(dY):
            model.attrs["last_dY"] = dY
            return dY

        return (
            TransformerModelOutput(
                outputs=[
                    [
                        Ragged(
                            model.ops.xp.ones((x.dataXd.shape[0], 2)), lengths=x.lengths
                        )
                    ]
                    for x in X
                ],
                last_layer_only=False,
            ),
            backprop,
        )

    return Model("mock_transformer", forward)


def test_with_non_ws_tokens(sample_docs_with_spaces, wordpiece_toy_encoder):
    mock_transformer = _mock_transformer()
    model = with_non_ws_tokens(
        chain(wordpiece_toy_encoder, mock_transformer, remove_bos_eos())
    )
    model.initialize()
    Y, backprop = model(sample_docs_with_spaces, is_train=True)

    yl0 = Y.all_outputs[0][0]
    y1_check = model.ops.xp.ones((15, 2))
    y1_check[6, :] = 0.0
    model.ops.xp.testing.assert_array_equal(yl0.dataXd, y1_check)

    yl1 = Y.all_outputs[1][0]
    y2_check = model.ops.xp.ones((17, 2))
    y2_check[4, :] = 0.0
    y2_check[8, :] = 0.0
    y2_check[16, :] = 0.0
    model.ops.xp.testing.assert_array_equal(yl1.dataXd, y2_check)

    dY = [
        [
            Ragged(
                model.ops.xp.arange(float(yl0.dataXd.size)).reshape(yl0.dataXd.shape),
                lengths=yl0.lengths,
            )
        ],
        [
            Ragged(
                model.ops.xp.arange(float(yl1.dataXd.size)).reshape(yl1.dataXd.shape),
                lengths=yl1.lengths,
            )
        ],
    ]

    backprop(dY)

    transformer_dY = mock_transformer.attrs["last_dY"]
    model.ops.xp.testing.assert_array_equal(
        transformer_dY[0][0].dataXd,
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
            [14.0, 15.0],
            [16.0, 17.0],
            [18.0, 19.0],
            [20.0, 21.0],
            [22.0, 23.0],
            [24.0, 25.0],
            [26.0, 27.0],
            [28.0, 29.0],
            [0.0, 0.0],
        ],
    )
    model.ops.xp.testing.assert_array_equal(
        transformer_dY[1][0].dataXd,
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
            [18.0, 19.0],
            [20.0, 21.0],
            [22.0, 23.0],
            [24.0, 25.0],
            [26.0, 27.0],
            [28.0, 29.0],
            [30.0, 31.0],
            [0.0, 0.0],
        ],
    )
