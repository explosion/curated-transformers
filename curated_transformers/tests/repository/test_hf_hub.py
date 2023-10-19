import pytest

from curated_transformers.repository import HfHubRepository

TEST_LINES = ["Line 1", "Line 2 🐍", "🤖" "\n\n", ""]


@pytest.mark.upload_tests
def test_hf_hub_upload_txt():
    repo = HfHubRepository("explosion-testing/hf-upload-test")

    write_file = repo.file("test.txt")
    str_to_write = "\n".join(TEST_LINES)
    with write_file.open(mode="w") as f:
        f.write(str_to_write)

    read_file = repo.file("test.txt")
    with read_file.open(mode="r", encoding="utf-8") as f:
        str_read = f.read()
        assert str_read == str_to_write


@pytest.mark.upload_tests
def test_hf_hub_upload_binary():
    repo = HfHubRepository("explosion-testing/hf-upload-test")

    write_file = repo.file("test.bin")
    bytes_to_write = b"\x0D\x0E\x0A\x0D\x0B\x0E\x0E\x0F"
    with write_file.open(mode="wb") as f:
        f.write(bytes_to_write)

    read_file = repo.file("test.bin")
    with read_file.open(mode="rb") as f:
        bytes_read = f.read()
        assert bytes_read == bytes_to_write


@pytest.mark.upload_tests
def test_hf_hub_transactions():
    repo = HfHubRepository("explosion-testing/hf-upload-test")

    with repo.transaction() as tx:
        with tx.open("inner/test.txt", mode="w", encoding="utf-8") as f:
            str_to_write = "\n".join(TEST_LINES)
            f.write(str_to_write)
        with tx.open("test.bin", mode="wb") as f:
            bytes_to_write = b"\x0D\x0E\x0A\x0D\x0B\x0E\x0E\x0F"
            f.write(bytes_to_write)

    with repo.file("inner/test.txt").open(mode="r", encoding="utf-8") as f:
        str_read = f.read()
        assert str_read == str_to_write
    with repo.file("test.bin").open(mode="rb") as f:
        bytes_read = f.read()
        assert bytes_read == bytes_to_write


def test_hf_hub_failures():
    repo = HfHubRepository("explosion-testing/hf-upload-test")
    write_file = repo.file("nonexistent.bin")

    assert write_file.path is None
    with pytest.raises(FileNotFoundError):
        with write_file.open("r") as f:
            pass

    with pytest.raises(OSError):
        with write_file.open("a") as f:
            pass

    with pytest.raises(OSError, match="mode 'a' is not supported"):
        with repo.transaction() as tx:
            f = tx.open("test.txt", mode="a", encoding="utf-8")

    with pytest.raises(OSError, match="mode 'rb' is not supported"):
        with repo.transaction() as tx:
            f = tx.open("test.bin", mode="rb")

    with pytest.raises(OSError, match="already been opened"):
        with repo.transaction() as tx:
            f = tx.open("test.bin", mode="w")
            f1 = tx.open("test.bin", mode="w")
