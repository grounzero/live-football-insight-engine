"""Typed helpers shared by the test suite.

Starlette's ``TestClient`` inherits its request methods from ``httpx.Client``
and annotates them against httpx's private ``_types`` module, so a strict
checker resolves the whole signature — and every response attribute reached
through it — as unknown. Starlette is itself deprecating this integration in
favour of httpx2, so the gap is upstream and temporary.

:class:`ApiClient` is a thin façade with the same method names, exposing exactly
the surface these tests use with real types. It keeps the single unavoidable
``cast`` in one documented place instead of scattering suppressions across every
assertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient


def approx(expected: float, *, rel: float | None = None, abs: float | None = None) -> Any:
    """``pytest.approx`` with a typed signature.

    pytest ships ``py.typed`` but leaves ``approx`` itself unannotated, so every
    call site reads as partially unknown. The return stays ``Any`` because the
    real object is a comparison proxy, not a float.
    """
    return pytest.approx(expected, rel=rel, abs=abs)  # pyright: ignore[reportUnknownMemberType]


class ApiResponse(Protocol):
    """The response surface these tests assert against."""

    @property
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    def json(self) -> Any: ...

    def iter_lines(self) -> Iterator[str]: ...


class ApiClient:
    """Typed façade over :class:`fastapi.testclient.TestClient`.

    Method names match the wrapped client so call sites read the same as they
    would against the real thing.
    """

    def __init__(self, client: TestClient) -> None:
        """Wrap a test client."""
        self._client = client

    def get(self, url: str, **kwargs: Any) -> ApiResponse:
        """Issue a GET request."""
        return cast("ApiResponse", self._client.get(url, **kwargs))  # pyright: ignore[reportUnknownMemberType]

    def post(self, url: str, **kwargs: Any) -> ApiResponse:
        """Issue a POST request."""
        return cast("ApiResponse", self._client.post(url, **kwargs))  # pyright: ignore[reportUnknownMemberType]

    def stream(self, method: str, url: str, **kwargs: Any) -> AbstractContextManager[ApiResponse]:
        """Open a streaming request as a context manager."""
        return cast(
            "AbstractContextManager[ApiResponse]",
            self._client.stream(method, url, **kwargs),  # pyright: ignore[reportUnknownMemberType]
        )
