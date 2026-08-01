"""Transparent rule-based fallback.

This is **not** a machine-learning model and is never reported as one. It exists
for three reasons:

* it makes the whole vertical path runnable before any model is trained, and
  keeps it runnable if artifacts are missing;
* it gives the service something safe to degrade to, clearly labelled, rather
  than emitting nothing at all;
* it is a floor to measure against — a temporal model that cannot beat four
  hand-written terms is not earning its complexity.

Its output is scored in a separate metrics namespace and excluded from every
model performance table. ``ModelMetadata.is_ml`` is ``False``, which the API
surfaces and the demo renders as "Fallback — not ML".
"""

from __future__ import annotations

from typing import Final

import numpy as np

from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.models.base import ModelMetadata, validate_batch

#: Coefficients of the logistic score. Chosen by hand from the geometry of the
#: problem, not fitted: a ball close to the box, carried goalward, with runners
#: beyond it and space to run into, is more likely to enter the penalty area.
#: They are deliberately round numbers so nobody mistakes them for trained
#: parameters.
BIAS: Final = -3.2
W_BALL_ADVANCEMENT: Final = 3.6
W_GOALWARD_SPEED: Final = 0.9
W_ATTACKERS_AHEAD: Final = 0.35
W_SPACE_AHEAD: Final = 0.55
W_DEFENSIVE_PRESSURE: Final = -0.45


class HeuristicPredictor:
    """Deterministic rule-based scorer over the final timestep of a window."""

    def __init__(
        self,
        threshold: float = 0.5,
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    ) -> None:
        """Initialise the fallback.

        Args:
            threshold: Decision threshold recorded in metadata.
            spec: Feature schema; used to resolve column indices by name.
        """
        self._spec = spec
        self._idx = {
            name: spec.index(name)
            for name in (
                "ball_x",
                "ball_v_goalward",
                "attackers_ahead_of_ball",
                "space_ahead_of_ball",
                "defenders_within_10m_of_ball",
                "is_dead_ball",
            )
        }
        self._metadata = ModelMetadata(
            name="heuristic-fallback",
            version="1.0.0",
            kind="heuristic",
            is_ml=False,
            feature_schema_hash=spec.schema_hash,
            sequence_length=0,  # accepts any sequence length; uses the last step
            n_features=spec.n_features,
            threshold=threshold,
            notes=(
                "Hand-written rule-based fallback. Not a trained model and not "
                "counted in model performance metrics."
            ),
        )

    @property
    def metadata(self) -> ModelMetadata:
        """Identity of this predictor."""
        return self._metadata

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Score windows from their most recent timestep.

        Args:
            windows: Array ``(batch, sequence, features)`` or a single ``(sequence,
                features)`` window.

        Returns:
            Probabilities of shape ``(batch,)``.
        """
        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3 or arr.shape[2] != self._spec.n_features:
            # Reuse the shared contract error for a consistent message.
            validate_batch(arr, self._metadata)
        last = arr[:, -1, :]

        ball_x = last[:, self._idx["ball_x"]]
        v_goal = np.clip(last[:, self._idx["ball_v_goalward"]], 0.0, 12.0) / 12.0
        ahead = np.clip(last[:, self._idx["attackers_ahead_of_ball"]], 0.0, 6.0) / 6.0
        space = np.clip(last[:, self._idx["space_ahead_of_ball"]], 0.0, 30.0) / 30.0
        pressure = np.clip(last[:, self._idx["defenders_within_10m_of_ball"]], 0.0, 6.0) / 6.0
        dead = last[:, self._idx["is_dead_ball"]]

        # ball_x is already normalised to [-1, 1]; map to advancement in [0, 1].
        advancement = np.clip((ball_x + 1.0) / 2.0, 0.0, 1.0)

        score = (
            BIAS
            + W_BALL_ADVANCEMENT * advancement
            + W_GOALWARD_SPEED * v_goal
            + W_ATTACKERS_AHEAD * ahead
            + W_SPACE_AHEAD * space
            + W_DEFENSIVE_PRESSURE * pressure
        )
        prob = 1.0 / (1.0 + np.exp(-score))
        # Nothing is imminent while the ball is out of play.
        return np.where(dead > 0.5, 0.0, prob).astype(np.float64)
