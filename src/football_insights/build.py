"""What build is running.

A deployment that cannot say which commit it came from has to be identified by
inference — comparing wording on the page against the source, or watching a
fingerprint change — and that goes wrong in the direction that matters: a slow
deploy and a failed one look identical, and the natural reading is the
pessimistic one. This module exists so the question has an answer.

The commit is baked in at image build time, because a running container has no
git and no repository to ask. Two sources, in order:

* ``FI_BUILD_COMMIT``, set from a Docker build argument, which works for any
  image however it was built;
* ``RAILWAY_GIT_COMMIT_SHA``, which the platform injects on its own.

Neither is required. A source-tree run reports the version and no commit, which
is honest: there is no build, and the code may be dirtier than any commit.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Final

from football_insights.types import JsonDict

#: Build argument the Dockerfile bakes in. Read at import, not at call time:
#: the value is fixed for the life of the process, and re-reading the
#: environment per request would invite something to change it.
_COMMIT_VARS: Final = ("FI_BUILD_COMMIT", "RAILWAY_GIT_COMMIT_SHA")

#: Short commits are what a person reads off a page and types into `git show`.
_SHORT_SHA_LENGTH: Final = 7


def _package_version() -> str:
    """Installed package version, or a marker when running from a source tree."""
    try:
        return version("football-insights")
    except PackageNotFoundError:  # pragma: no cover - only on a bare checkout
        return "unknown"


def _commit() -> str | None:
    """The commit this build came from, if it was recorded."""
    for name in _COMMIT_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value[:_SHORT_SHA_LENGTH]
    return None


def build_info() -> JsonDict:
    """Version and commit of the running build.

    Returns:
        ``version`` always; ``commit`` only when a build recorded one, so an
        absent key means "not built from a known commit" rather than being
        confused with a commit whose value happens to be empty.
    """
    info: JsonDict = {"version": _package_version()}
    commit = _commit()
    if commit is not None:
        info["commit"] = commit
    return info
