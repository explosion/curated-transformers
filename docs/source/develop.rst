Development
===========

Branches
--------

We use two branches during regular development. If the current version is
*1.0.3*, then the active branches are:

- ``main``: the development branch that will lead up to *1.1.0*.
- ``v1.0.x``: the bugfix branch that will lead up to *1.0.4*.

Following semver, only bug fixes must be pushed to ``v1.0.x``. When applicable,
a bug should first be fixed in the ``main`` branch using a PR. After that, a
backport PR should be made for the ``v1.0.x`` branch with the *backport* label.
