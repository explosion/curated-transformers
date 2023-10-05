import catalogue
from catalogue import Registry


class registry(object):
    """
    Registry for models. These registries are used by auto classes to
    discover the available models.
    """

    causal_lms: Registry = catalogue.create(
        "curated_transformers", "causal_lms", entry_points=True
    )
    decoders: Registry = catalogue.create(
        "curated_transformers", "decoders", entry_points=True
    )
    encoders: Registry = catalogue.create(
        "curated_transformers", "encoders", entry_points=True
    )
