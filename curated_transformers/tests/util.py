import contextlib
from pathlib import Path
import shutil
import tempfile


@contextlib.contextmanager
def make_tempdir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d))
