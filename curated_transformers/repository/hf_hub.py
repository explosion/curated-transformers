import warnings
from typing import Optional

import huggingface_hub
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests import HTTPError, ReadTimeout  # type: ignore

from ..repository.file import LocalFile, RepositoryFile
from .repository import Repository


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
        try:
            return LocalFile(
                path=hf_hub_download(
                    repo_id=self.name, filename=path, revision=self.revision
                )
            )
        except (
            EntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        ) as e:
            raise FileNotFoundError(f"File not found: {self.pretty_path(path)}") from e
        except Exception as e:
            raise OSError(f"File could not be opened: {self.pretty_path(path)}") from e

    def pretty_path(self, path: Optional[str] = None) -> str:
        if not path:
            return f"{self.name} (revision: {self.revision})"
        return f"{self.name}/{path} (revision: {self.revision})"


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
        resolved = huggingface_hub.try_to_load_from_cache(
            repo_id=repo_id, filename=filename, revision=revision
        )
        if resolved is None or resolved is huggingface_hub._CACHED_NO_EXIST:
            # Not found, rethrow.
            raise e
        else:
            warnings.warn(
                f"Couldn't reach Hugging Face Hub; using cached artifact for '{repo_id}@{revision}:{filename}'"
            )
    return resolved
