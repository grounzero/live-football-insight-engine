"""Feature schema and its version hash.

The serving path refuses to run a model whose recorded schema hash differs from
the one the running code produces. That turns a silent feature-order mismatch —
the sort of bug that quietly degrades a live model for weeks — into a refusal to
become ready.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from football_insights.types import JsonDict

#: Ordered feature names. Order is part of the contract: changing it changes the
#: schema hash and invalidates existing model artifacts, which is the intent.
#:
#: Grouped by what they describe. Every feature is invariant to player identity
#: and to which end of the pitch is being attacked: coordinates are rotated so
#: the team in possession always attacks +x.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    # --- ball state -------------------------------------------------------
    "ball_x",
    "ball_y_abs",
    "ball_speed",
    "ball_v_goalward",
    "ball_v_lateral",
    "ball_dist_to_goal",
    "ball_goal_angle",
    "ball_in_final_third",
    "ball_in_box",
    # --- attacking shape --------------------------------------------------
    "attackers_ahead_of_ball",
    "attackers_in_box",
    "attackers_in_final_third",
    "attack_centroid_x",
    "attack_width",
    "attack_depth",
    "attack_highest_x",
    "attack_mean_v_goalward",
    "attackers_moving_goalward",
    # --- defending shape --------------------------------------------------
    "defenders_in_box",
    "defenders_goalside_of_ball",
    "defensive_line_x",
    "defensive_compactness",
    "defence_centroid_x",
    "defence_mean_v_goalward",
    "gk_dist_to_ball",
    "gk_dist_to_goal",
    # --- contest around the ball -----------------------------------------
    "nearest_defender_dist",
    "second_nearest_defender_dist",
    "mean_three_nearest_defender_dist",
    "nearest_attacker_support_dist",
    "space_ahead_of_ball",
    "defenders_within_10m_of_ball",
    # --- possession context (causal, from CausalEventView) ---------------
    "possession_duration_s",
    "possession_event_count",
    "is_dead_ball",
    "event_in_flight",
    "recent_pass_count",
    "recent_box_entry_count",
    "time_since_last_box_entry",
)

#: Bumped by hand when the *meaning* of a feature changes without its name
#: changing, which the hash alone could not detect.
#:
#: 2 — attacking shape aggregates (``attack_centroid_x``, ``attack_width``,
#:     ``attack_depth``, ``attack_highest_x``) now exclude the attacking
#:     goalkeeper. Names were unchanged, so the hash alone would not have
#:     invalidated existing artifacts; this is exactly the case the revision
#:     exists for.
FEATURE_SCHEMA_REVISION: Final = 2


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """An immutable description of the model input contract."""

    names: tuple[str, ...] = FEATURE_NAMES
    revision: int = FEATURE_SCHEMA_REVISION

    def __post_init__(self) -> None:
        """Reject duplicate feature names, which would make indices ambiguous."""
        if len(set(self.names)) != len(self.names):
            seen: set[str] = set()
            dupes = sorted({n for n in self.names if n in seen or seen.add(n)})  # type: ignore[func-returns-value]
            msg = f"duplicate feature names in spec: {dupes}"
            raise ValueError(msg)

    @property
    def n_features(self) -> int:
        """Number of features per timestep."""
        return len(self.names)

    @property
    def schema_hash(self) -> str:
        """Stable short hash over the ordered names and the revision."""
        payload = f"v{self.revision}:" + ",".join(self.names)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def index(self, name: str) -> int:
        """Column index of a named feature.

        Args:
            name: Feature name.

        Returns:
            Its position in the feature vector.

        Raises:
            KeyError: If the feature is not part of this schema.
        """
        try:
            return self.names.index(name)
        except ValueError as exc:
            msg = f"unknown feature {name!r}"
            raise KeyError(msg) from exc

    def describe(self) -> JsonDict:
        """Serialisable description recorded in model metadata."""
        return {
            "schema_hash": self.schema_hash,
            "revision": self.revision,
            "n_features": self.n_features,
            "names": list(self.names),
        }


DEFAULT_FEATURE_SPEC: Final = FeatureSpec()
