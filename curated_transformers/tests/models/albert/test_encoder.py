import pytest

from curated_transformers.models.albert import AlbertConfig, AlbertEncoder


def test_rejects_incorrect_number_of_groups():
    config = AlbertConfig(num_hidden_groups=5)
    with pytest.raises(ValueError, match=r"must be divisable"):
        AlbertEncoder(config)
