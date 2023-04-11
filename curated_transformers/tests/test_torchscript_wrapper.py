import pytest
import numpy
from torch.nn import Linear
from thinc.layers import (
    PyTorchWrapper_v2,
    TorchScriptWrapper_v1,
    pytorch_to_torchscript_wrapper,
)
from thinc.api import get_array_module


@pytest.mark.parametrize("nN,nI,nO", [(2, 3, 4)])
def test_pytorch_script(nN, nI, nO):
    model = PyTorchWrapper_v2(Linear(nI, nO)).initialize()

    script_model = pytorch_to_torchscript_wrapper(model)

    X = numpy.random.randn(nN, nI).astype("f")
    Y = model.predict(X)
    Y_script = script_model.predict(X)
    xp = get_array_module(Y)
    xp.testing.assert_array_equal(Y, Y_script)

    serialized = script_model.to_bytes()
    script_model2 = TorchScriptWrapper_v1()
    script_model2.from_bytes(serialized)

    xp.testing.assert_array_equal(Y, script_model2.predict(X))
