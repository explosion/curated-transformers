import contextlib
from pathlib import Path
import shutil
import torch
import tempfile


@contextlib.contextmanager
def make_tempdir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d))


# Wrapper around torch.testing.assert_close with custom tolerances.
def torch_assertclose(
    a: torch.tensor, b: torch.tensor, *, atol: float = 1e-05, rtol: float = 1e-05
):
    torch.testing.assert_close(
        a,
        b,
        atol=atol,
        rtol=rtol,
    )
