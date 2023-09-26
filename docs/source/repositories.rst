Repositories
============

Models and tokenizers can be loaded from repositories using the ``from_repo``
method. You can add your own type of repository by implementing the
:py:class:`curated_transformers.repository.Repository` base class.

Base Classes
------------

.. autoclass:: curated_transformers.repository.Repository
   :members:
   :show-inheritance:

.. autoclass:: curated_transformers.repository.RepositoryFile
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
