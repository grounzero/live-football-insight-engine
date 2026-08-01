"""Predictor interface and model metadata.

Every scorer in the system — the heuristic fallback, the baselines and the
temporal model — satisfies :class:`Predictor`, so the service, the editorial
layer and the evaluation harness are written once.

:class:`ModelMetadata` travels with the artifact and is echoed by the service.
It carries ``is_ml`` explicitly: the fallback is a legitimate part of the system
but must never be mistaken for the trained model, in the API, the metrics or a
screenshot of the demo.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from football_insights.errors import SchemaVersionError
from football_insights.types import JsonDict


def git_revision() -> str | None:
    """Current git commit, or ``None`` outside a repository.

    Recorded in model metadata so a set of results can be traced to the code
    that produced it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = out.stdout.strip()
    return revision or None


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Everything needed to identify a model and judge whether to trust it."""

    name: str
    version: str
    kind: str
    is_ml: bool
    feature_schema_hash: str
    sequence_length: int
    n_features: int
    threshold: float = 0.5
    trained_at: str | None = None
    git_revision: str | None = None
    dataset_fingerprint: str | None = None
    config_fingerprint: str | None = None
    training_matches: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def now(cls, **kwargs: object) -> ModelMetadata:
        """Build metadata stamped with the current time and git revision."""
        kwargs.setdefault("trained_at", datetime.now(UTC).isoformat(timespec="seconds"))
        kwargs.setdefault("git_revision", git_revision())
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_dict(self) -> JsonDict:
        """Serialisable form written next to the artifact."""
        return asdict(self)

    def write(self, path: Path) -> None:
        """Write metadata as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> ModelMetadata:
        """Read metadata from JSON."""
        raw = json.loads(Path(path).read_text())
        raw["training_matches"] = tuple(raw.get("training_matches", ()))
        return cls(**raw)

    def require_schema(self, schema_hash: str) -> None:
        """Refuse to run against a mismatched feature schema.

        Args:
            schema_hash: Hash produced by the running code.

        Raises:
            SchemaVersionError: If the hashes differ.
        """
        if self.feature_schema_hash != schema_hash:
            msg = (
                f"model {self.name}:{self.version} was trained against feature schema "
                f"{self.feature_schema_hash} but this build produces {schema_hash}. "
                "Refusing to serve: the feature order or meaning has changed."
            )
            raise SchemaVersionError(msg)


@runtime_checkable
class Predictor(Protocol):
    """Anything that scores observation windows.

    Implementations must be deterministic: identical input gives identical
    output, which the service and the test suite both rely on.
    """

    @property
    def metadata(self) -> ModelMetadata:
        """Identity and provenance of this predictor."""
        ...

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Score a batch of observation windows.

        Args:
            windows: Array ``(batch, sequence_length, n_features)``.

        Returns:
            Probabilities of shape ``(batch,)`` in ``[0, 1]``.
        """
        ...


def validate_batch(windows: np.ndarray, metadata: ModelMetadata) -> np.ndarray:
    """Check a batch against the model's declared input contract.

    Args:
        windows: Candidate input; a single window ``(T, F)`` is accepted and
            promoted to a batch of one.
        metadata: The model's metadata.

    Returns:
        The batch as ``float32`` with shape ``(batch, T, F)``.

    Raises:
        ValueError: If the shape does not match the contract.
    """
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        msg = f"expected (batch, sequence, features), got shape {arr.shape}"
        raise ValueError(msg)
    if arr.shape[1] != metadata.sequence_length or arr.shape[2] != metadata.n_features:
        msg = (
            f"model {metadata.name} expects "
            f"({metadata.sequence_length}, {metadata.n_features}) per sample, "
            f"got ({arr.shape[1]}, {arr.shape[2]})"
        )
        raise ValueError(msg)
    if not np.isfinite(arr).all():
        msg = "input contains non-finite values; the window validator should have rejected it"
        raise ValueError(msg)
    return arr
