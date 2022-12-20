import pytest
import numpy
from torch.nn import Linear
from thinc.layers import (
    PyTorchWrapper_v2,
    TorchScriptWrapper_v1,
    pytorch_to_torchscript_wrapper,
)


@pytest.mark.parametrize("nN,nI,nO", [(2, 3, 4)])
def test_pytorch_script(nN, nI, nO):

    model = PyTorchWrapper_v2(Linear(nI, nO)).initialize()

    script_model = pytorch_to_torchscript_wrapper(model)

    X = numpy.random.randn(nN, nI).astype("f")
    Y = model.predict(X)
    Y_script = script_model.predict(X)
    numpy.testing.assert_allclose(Y, Y_script)

    serialized = script_model.to_bytes()
    script_model2 = TorchScriptWrapper_v1()
    script_model2.from_bytes(serialized)

    numpy.testing.assert_allclose(Y, script_model2.predict(X))
