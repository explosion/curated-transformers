import warnings
from importlib.util import find_spec

_has_scipy = find_spec("scipy") is not None and find_spec("scipy.stats") is not None
has_bitsandbytes = find_spec("bitsandbytes") is not None

# As of v0.40.0, `bitsandbytes` doesn't correctly specify `scipy` as an installation
# dependency. This can lead to situations where the former is installed but
# the latter isn't and the ImportError gets masked. So, we additionally check
# for the presence of `scipy`.
if has_bitsandbytes and not _has_scipy:
    warnings.warn(
        "The `bitsandbytes` library is installed but its dependency "
        "`scipy` isn't. Please install `scipy` to correctly load `bitsandbytes`."
    )
    has_bitsandbytes = False

has_safetensors = find_spec("safetensors")
