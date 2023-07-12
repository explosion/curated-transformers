import warnings

try:
    import transformers

    has_hf_transformers = True
except ImportError:
    transformers = None  # type: ignore
    has_hf_transformers = False


def _check_scipy_presence() -> bool:
    try:
        from scipy.stats import norm

        return True
    except ImportError:
        return False


def _check_bnb_presence() -> bool:
    try:
        import bitsandbytes
    except ModuleNotFoundError:
        return False
    except ImportError:
        pass
    return True


# As of v0.40.0, `bitsandbytes` doesn't correctly specify `scipy` as an installation
# dependency. This can lead to situations where the former is installed but
# the latter isn't and the ImportError gets masked. So, we additionally check
# for the presence of `scipy`.
if _check_scipy_presence():
    try:
        import bitsandbytes

        has_bitsandbytes = True
    except ImportError:
        bitsandbytes = None  # type: ignore
        has_bitsandbytes = False
else:
    if _check_bnb_presence():
        warnings.warn(
            "The `bitsandbytes` library is installed but its dependency "
            "`scipy` isn't. Please install `scipy` to correctly load `bitsandbytes`."
        )
    bitsandbytes = None  # type: ignore
    has_bitsandbytes = False
