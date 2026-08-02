"""Non-neural baselines.

Two of them, both trained and evaluated on exactly the same splits, windows and
metrics as the temporal model. That symmetry is the point: without it, a
neural network's numbers cannot be read as evidence that the architecture is
earning its complexity.

Sequences are reduced to a fixed-length summary — the final timestep plus the
mean, spread and net change across the window. The final timestep carries the
current state, and the three aggregates carry how it got there, which is most of
what a five-second window contains.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.models.base import ModelMetadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sklearn.base import BaseEstimator

BaselineKind = Literal["logistic", "gbdt"]

#: Names of the aggregate blocks, in the order they are concatenated.
AGGREGATES = ("last", "mean", "std", "delta")


def summarise_windows(windows: np.ndarray) -> np.ndarray:
    """Reduce sequences to fixed-length summaries.

    Args:
        windows: Array ``(n, sequence_length, n_features)``.

    Returns:
        Array ``(n, 4 * n_features)``.
    """
    arr = np.asarray(windows, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return np.concatenate(
        [arr[:, -1, :], arr.mean(axis=1), arr.std(axis=1), arr[:, -1, :] - arr[:, 0, :]],
        axis=1,
    )


def summary_feature_names(spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> list[str]:
    """Names for the summarised features, used in coefficient reports."""
    return [f"{block}__{name}" for block in AGGREGATES for name in spec.names]


class BaselinePredictor:
    """A scikit-learn classifier wrapped in the :class:`Predictor` interface."""

    def __init__(
        self,
        estimator: BaseEstimator,
        metadata: ModelMetadata,
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    ) -> None:
        """Wrap a fitted estimator.

        Args:
            estimator: A fitted classifier exposing ``predict_proba``.
            metadata: Model metadata.
            spec: Feature schema.
        """
        self._estimator = estimator
        self._metadata = metadata
        self._spec = spec

    @property
    def metadata(self) -> ModelMetadata:
        """Identity and provenance."""
        return self._metadata

    @property
    def estimator(self) -> BaseEstimator:
        """The underlying scikit-learn estimator."""
        return self._estimator

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Score a batch of windows.

        Args:
            windows: Array ``(n, sequence_length, n_features)``.

        Returns:
            Probabilities of shape ``(n,)``.
        """
        summary = summarise_windows(windows)
        return np.asarray(self._estimator.predict_proba(summary))[:, 1]

    def save(self, path: Path) -> None:
        """Persist the estimator and its metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"estimator": self._estimator, "metadata": self._metadata}, handle)

    @classmethod
    def load(cls, path: Path, spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> BaselinePredictor:
        """Load a persisted baseline.

        Args:
            path: Artifact path.
            spec: Feature schema the running code produces.

        Returns:
            The loaded predictor.

        Raises:
            SchemaVersionError: If the artifact was built against a different
                feature schema.
        """
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        metadata: ModelMetadata = payload["metadata"]
        metadata.require_schema(spec.schema_hash)
        return cls(payload["estimator"], metadata, spec)


def build_estimator(kind: BaselineKind, seed: int, class_weight: float | None) -> BaseEstimator:
    """Construct an unfitted baseline.

    Args:
        kind: Which baseline to build.
        seed: Random seed.
        class_weight: Weight for the positive class; ``None`` uses balanced
            weighting derived from the data.

    Returns:
        The estimator.
    """
    weights: Mapping[int, float] | Literal["balanced"] = (
        "balanced" if class_weight is None else {0: 1.0, 1: class_weight}
    )
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        C=0.5,
                        class_weight=weights,
                        random_state=seed,
                    ),
                ),
            ]
        )
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_depth=5,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
        class_weight=weights,
    )


def train_baseline(
    kind: BaselineKind,
    windows: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    class_weight: float | None = None,
    training_matches: tuple[str, ...] = (),
    dataset_fingerprint: str | None = None,
    config_fingerprint: str | None = None,
) -> BaselinePredictor:
    """Fit a baseline on summarised windows.

    Args:
        kind: Which baseline to fit.
        windows: Training windows ``(n, sequence_length, n_features)``.
        labels: Binary labels.
        seed: Random seed.
        spec: Feature schema.
        class_weight: Positive-class weight; balanced when ``None``.
        training_matches: Matches used, recorded in metadata.
        dataset_fingerprint: Dataset hash, recorded in metadata.
        config_fingerprint: Config hash, recorded in metadata.

    Returns:
        The fitted predictor.
    """
    estimator = build_estimator(kind, seed, class_weight)
    estimator.fit(summarise_windows(windows), labels)
    metadata = ModelMetadata.now(
        name=f"baseline-{kind}",
        version="1.0.0",
        kind=kind,
        is_ml=True,
        feature_schema_hash=spec.schema_hash,
        sequence_length=int(windows.shape[1]),
        n_features=int(windows.shape[2]),
        training_matches=training_matches,
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config_fingerprint,
        notes=f"{kind} on last/mean/std/delta summaries of the observation window",
    )
    return BaselinePredictor(estimator, metadata, spec)


def logistic_coefficients(
    predictor: BaselinePredictor, spec: FeatureSpec = DEFAULT_FEATURE_SPEC, top: int = 15
) -> list[tuple[str, float]]:
    """Largest-magnitude logistic-regression coefficients.

    Only meaningful for the logistic baseline, whose inputs are standardised so
    coefficients are comparable across features.

    Args:
        predictor: A fitted logistic baseline.
        spec: Feature schema.
        top: How many coefficients to return.

    Returns:
        ``(name, coefficient)`` pairs ordered by absolute value.
    """
    estimator = predictor.estimator
    model = estimator.named_steps["model"] if isinstance(estimator, Pipeline) else estimator
    coefficients = np.asarray(model.coef_).ravel()
    names = summary_feature_names(spec)
    order = np.argsort(np.abs(coefficients))[::-1][:top]
    return [(names[i], round(float(coefficients[i]), 4)) for i in order]
