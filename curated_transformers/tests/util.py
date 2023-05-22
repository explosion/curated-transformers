import contextlib
from pathlib import Path
import shutil
import torch
import tempfile

from curated_transformers._compat import transformers


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


def compare_tokenizer_outputs_with_hf_tokenizer(sample_texts, hf_name, tokenizer_cls):
    tokenizer = tokenizer_cls.from_hf_hub(name=hf_name)
    pieces = tokenizer(sample_texts)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_name)
    hf_pieces = hf_tokenizer(sample_texts)

    assert pieces.ids == hf_pieces["input_ids"]

    decoded = tokenizer.decode(pieces.ids)
    hf_decoded = [
        hf_tokenizer.decode(ids, skip_special_tokens=True)
        for ids in hf_pieces["input_ids"]
    ]

    assert decoded == hf_decoded
