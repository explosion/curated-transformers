Repositories
============

Models and tokenizers can be loaded from repositories using the ``from_repo``
method. You can add your own type of repository by implementing the
:py:class:`curated_transformers.repository.Repository` base class.

This is an example repository that opens files on the local filesystem:

.. code-block:: python

   import os.path
   from typing import Optional

   from curated_transformers.repository import Repository, RepositoryFile

   class LocalRepository(Repository):
      def __init__(self, path: str):
         super().__init__()
         self.repo_path = path

      def file(self, path: str) -> RepositoryFile:
         full_path = f"{self.repo_path}/path"
         if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
         return LocalFile(path=full_path)

      def pretty_path(self, path: Optional[str] = None) -> str:
         return self.full_path

Base Classes
------------

.. autoclass:: curated_transformers.repository.Repository
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.repository.RepositoryFile
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.repository.TransactionContext
   :members:
   :show-inheritance:   

Repositories
------------

.. autoclass:: curated_transformers.repository.FsspecRepository
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.repository.HfHubRepository
   :members:
   :show-inheritance:

Repository Files
----------------

.. autoclass:: curated_transformers.repository.FsspecFile
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.repository.LocalFile
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.repository.HfHubFile
   :members:
   :show-inheritance: