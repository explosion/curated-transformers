import os
import warnings
from tempfile import NamedTemporaryFile
from typing import IO, Any, AnyStr, Dict, Iterator, List, Optional, Union

import huggingface_hub
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests import HTTPError, ReadTimeout  # type: ignore
from typing_extensions import Buffer

from ..repository.file import LocalFile, RepositoryFile
from .repository import Repository
from .transaction import TransactionContext


class HfHubFile(RepositoryFile):
    """
    Wraps either a remote file on a Hugging Face Hub repository
    or a local file in the Hugging Face cache.
    """

    def __init__(self, repo: "HfHubRepository", path: str):
        """
        Construct a Hugging Face file representation.

        :param repo:
            The parent repository.
        :param path:
            The path of the remote file in the repository.
        """
        super().__init__()

        self._repo = repo
        self._remote_path = path
        self._cached_path = lookup_file_in_cache(repo.name, repo.revision, path)

    def _validate_model_repo(self):
        api = HfApi()
        # This will raise RepositoryNotFoundError/RevisionNotFoundError upon failure.
        _ = api.model_info(repo_id=self._repo.name, revision=self._repo.revision)

    def _download_to_cache(self) -> LocalFile:
        local_file = LocalFile(
            hf_hub_download(
                repo_id=self._repo.name,
                filename=self._remote_path,
                revision=self._repo.revision,
            )
        )
        self._cached_path = local_file.path
        return local_file

    def open(self, mode: str = "rb", encoding: Optional[str] = None) -> IO:
        try:
            if "r" in mode:
                # Attempt to download the file if we want to read from it.
                local_file = self._download_to_cache()
                return local_file.open(mode=mode, encoding=encoding)
            elif "w" in mode:
                # Return a proxy for the remote file.
                self._validate_model_repo()
                return UploadStagingBuffer(
                    self._repo, self._remote_path, mode=mode, encoding=encoding
                )
            else:
                raise OSError(
                    f"Unsupported mode '{mode}' for opening Hugging Face Hub file"
                )
        except (
            EntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        ) as e:
            raise FileNotFoundError(
                f"File not found: {self._repo.pretty_path(self._remote_path)}"
            ) from e
        except Exception as e:
            raise OSError(
                f"File could not be opened: {self._repo.pretty_path(self._remote_path)}"
            ) from e

    @property
    def path(self) -> Optional[str]:
        return self._cached_path

    def exists(self) -> bool:
        try:
            return self._download_to_cache().exists()
        except:
            return False


class HfHubRepository(Repository):
    """
    Hugging Face Hub repository.
    """

    def __init__(self, name: str, *, revision: str = "main"):
        """
        :param name:
            Name of the repository on Hugging Face Hub.
        :param revision:
            Source repository revision. Can either be a branch name
            or a SHA hash of a commit.
        """
        super().__init__()
        self.name = name
        self.revision = revision

    def file(self, path: str) -> RepositoryFile:
        return HfHubFile(repo=self, path=path)

    def pretty_path(self, path: Optional[str] = None) -> str:
        if not path:
            return f"{self.name} (revision: {self.revision})"
        return f"{self.name}/{path} (revision: {self.revision})"

    def transaction(self) -> TransactionContext:
        return HfHubTransactionContext(self)


class HfHubTransactionContext(TransactionContext):
    """
    Hugging Face Hub transaction context manager. Files opened
    for writing are backed by temporary files on the host and
    uploaded to the Hub when the transaction completes successfully.
    """

    def __init__(self, repo: HfHubRepository):
        """
        :param repo:
            The parent repository.
        """
        super().__init__()
        self._repo = repo
        self._file_mappings: Dict[str, IO] = (
            {}
        )  # Maps remote file paths to local temporary files

    def open(self, path: str, mode: str, encoding: Optional[str] = None) -> IO:
        if path in self._file_mappings:
            raise OSError(f"Path '{path}' has already been opened")
        elif "w" not in mode:
            raise OSError(
                f"File open mode '{mode}' is not supported inside a transaction - "
                "files can only be opened for writing"
            )

        temp_file = NamedTemporaryFile(mode, encoding=encoding, delete=False)
        self._file_mappings[path] = temp_file
        return temp_file

    @property
    def repo(self) -> Repository:
        return self._repo

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            if exc_type is None:
                self._upload_temp_files()
        finally:
            self._release_temp_files()

    def _upload_temp_files(self):
        if len(self._file_mappings) == 0:
            return

        # Flush buffers.
        for temp_file in self._file_mappings.values():
            if not temp_file.closed:
                temp_file.flush()

        # Collate all files as a single commit.
        commit_ops = [
            CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=temp_file.name)
            for remote_path, temp_file in self._file_mappings.items()
        ]
        filenames = "\n".join(self._file_mappings.keys())
        commit_message = f"Uploading files:\n{filenames}"

        api = HfApi()
        _ = api.create_commit(
            repo_id=self._repo.name,
            revision=self._repo.revision,
            operations=commit_ops,
            commit_message=commit_message,
            run_as_future=False,
        )

    def _release_temp_files(self):
        for temp_file in self._file_mappings.values():
            try:
                if not temp_file.closed:
                    temp_file.close()
            finally:
                os.remove(temp_file.name)


class UploadStagingBuffer(IO):
    """
    Represents a local temporary file buffer that gets uploaded
    to a repository when it's closed.
    """

    def __init__(
        self,
        repo: HfHubRepository,
        remote_path: str,
        *,
        mode: str,
        encoding: Optional[str],
    ):
        super().__init__()
        self._repo = repo
        self._remote_path = remote_path
        self._temp_file = NamedTemporaryFile(mode, encoding=encoding, delete=False)

    def _upload(self):
        try:
            api = HfApi()
            result = api.upload_file(
                path_in_repo=self._remote_path,
                path_or_fileobj=self._temp_file,
                repo_id=self._repo.name,
                revision=self._repo.revision,
                run_as_future=False,
            )
        except Exception as e:
            raise OSError(
                f"Failed to upload to remote file on Hugging Face Hub @ {self._repo.pretty_path(self._remote_path)}.\n"
                f"Error: {e}"
            )

    # IO overrides follow.

    def __iter__(self) -> Iterator[Any]:
        return self._temp_file.__iter__()

    def __next__(self) -> Any:
        return self._temp_file.__next__()

    @property
    def mode(self) -> str:
        return self._temp_file.mode

    @property
    def name(self) -> str:
        return self._temp_file.name

    def close(self) -> None:
        return self._temp_file.close()

    @property
    def closed(self) -> bool:
        return self._temp_file.closed

    def fileno(self) -> int:
        return self._temp_file.fileno()

    def flush(self) -> None:
        return self._temp_file.flush()

    def isatty(self) -> bool:
        return self._temp_file.isatty()

    def read(self, n: int = -1) -> AnyStr:
        return self._temp_file.read(n)

    def readable(self) -> bool:
        return self._temp_file.readable()

    def readline(self, limit: int = -1) -> AnyStr:
        return self._temp_file.readline(limit)

    def readlines(self, hint: int = -1) -> List[AnyStr]:
        return self._temp_file.readlines(hint)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._temp_file.seek(offset, whence)

    def seekable(self) -> bool:
        return self._temp_file.seekable()

    def tell(self) -> int:
        return self._temp_file.tell()

    def truncate(self, size: int = None) -> int:  # type:ignore
        return self._temp_file.truncate(size)

    def writable(self) -> bool:
        return self._temp_file.writable()

    def write(self, s: Union[Any, Buffer, str]) -> int:
        return self._temp_file.write(s)

    def writelines(self, lines: List[AnyStr]) -> None:  # type:ignore
        self._temp_file.writelines(lines)

    def __enter__(self):
        return self._temp_file

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            if exc_type is not None:
                self._upload()
        finally:
            os.remove(self._temp_file.name)


def hf_hub_download(repo_id: str, filename: str, revision: str) -> str:
    """
    Resolve the provided filename and repository to a local file path. If the file
    is not present in the cache, it will be downloaded from the Hugging Face Hub.

    :param repo_id:
        Identifier of the source repository on Hugging Face Hub.
    :param filename:
        Name of the file in the source repository to download.
    :param revision:
        Source repository revision. Can either be a branch name
        or a SHA hash of a commit.
    :returns:
        Resolved absolute filepath.
    """

    # The HF Hub library's `hf_hub_download` function will always attempt to connect to the
    # remote repo and fetch its metadata even if it's locally cached (in order to update the
    # out-of-date artifacts in the cache). This can occasionally lead to `HTTPError/ReadTimeout`s if the
    # remote host is unreachable. Instead of failing loudly, we'll add a fallback that checks
    # the local cache for the artifacts and uses them if found.
    try:
        resolved = huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision
        )
    except (HTTPError, ReadTimeout) as e:
        # Attempt to check the cache.
        resolved = lookup_file_in_cache(repo_id, revision, filename)
        if resolved is None:
            # Not found, rethrow.
            raise e
        else:
            warnings.warn(
                f"Couldn't reach Hugging Face Hub; using cached artifact for '{repo_id}@{revision}:{filename}'"
            )
    return resolved


def lookup_file_in_cache(repo_id: str, revision: str, filename: str) -> Optional[str]:
    """
    Look up the file in the local Hugging Face Hub cache.

    :param repo_id:
        Identifier of the source repository on Hugging Face Hub.
    :param revision:
        Source repository revision. Can either be a branch name
        or a SHA hash of a commit.
    :param filename:
        Name of the file in the source repository to lookup in the cache.
    :returns:
        Local file path if found in cache, ``None`` otherwise.
    """

    resolved = huggingface_hub.try_to_load_from_cache(
        repo_id=repo_id, filename=filename, revision=revision
    )
    if resolved is None or resolved is huggingface_hub._CACHED_NO_EXIST:
        return None
    else:
        return resolved
