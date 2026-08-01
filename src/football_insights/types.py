"""Shared type aliases."""

from __future__ import annotations

from typing import Any, TypeAlias

#: A JSON-shaped document: reports, manifests and serialised metadata.
#:
#: ``dict[str, object]`` would be more precise about immutability but makes every
#: consumer cast before indexing, which buys nothing for structures that are
#: written straight to JSON and read back untyped.
JsonDict: TypeAlias = dict[str, Any]
