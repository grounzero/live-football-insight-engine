"""Types spanning the prediction and editorial stages.

The pipeline is deliberately two-stage:

``Prediction`` -> threshold -> ``InsightCandidate`` -> editorial -> ``Insight`` | ``Suppressed``

A statistically valid prediction is not automatically something a viewer should
be shown. Keeping the stages in separate types means the boundary cannot be
blurred by accident, and lets each be measured on its own terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from football_insights.types import JsonDict


class SuppressionReason(StrEnum):
    """Closed set of reasons a candidate did not reach a viewer.

    Every value is exported as a Prometheus label, so the set is fixed: a new
    reason is a deliberate change, not an ad-hoc string.
    """

    LOW_CONFIDENCE = "low_confidence"
    INVALID_WINDOW = "invalid_window"
    INSUFFICIENT_FRAMES = "insufficient_frames"
    COOLDOWN = "cooldown"
    DUPLICATE_RECENT = "duplicate_recent"
    STALE_SITUATION = "stale_situation"
    ALREADY_IN_BOX = "already_in_box"
    MODEL_UNAVAILABLE = "model_unavailable"
    SCHEMA_MISMATCH = "schema_mismatch"
    NOT_YET_SUSTAINED = "not_yet_sustained"
    DEAD_BALL = "dead_ball"


class InsightKind(StrEnum):
    """What an insight is about.

    Kinds are coarse on purpose. Each maps to one template family and carries
    its own cooldown, so a rising-threat message does not block a
    sustained-pressure message.
    """

    BUILDING_THREAT = "building_threat"
    ELEVATED_ENTRY_CHANCE = "elevated_entry_chance"
    SUSTAINED_PRESSURE = "sustained_pressure"


@dataclass(frozen=True, slots=True)
class Prediction:
    """A model's output for one observation window.

    Attributes:
        probability: Model output in ``[0, 1]``.
        window_valid: Whether the window was structurally sound. An invalid
            window still produces a :class:`Prediction` record for metrics, but
            can never yield an insight.
        invalid_reason: Populated when ``window_valid`` is ``False``.
    """

    probability: float
    match_time_s: float
    period: int
    attacking_team: str
    model_name: str
    model_version: str
    is_ml: bool
    window_valid: bool = True
    invalid_reason: str | None = None
    inference_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ContextFact:
    """A factual observation drawn from the same window as the prediction.

    Facts are measured, not predicted, so they can be stated plainly. Keeping
    them separate from the hedged headline is what lets an insight be both
    honest about uncertainty and concretely informative.
    """

    key: str
    text: str
    value: float


@dataclass(frozen=True, slots=True)
class InsightCandidate:
    """A prediction that cleared the threshold, before editorial review."""

    kind: InsightKind
    prediction: Prediction
    facts: tuple[ContextFact, ...] = ()

    @property
    def probability(self) -> float:
        """Model probability behind this candidate."""
        return self.prediction.probability

    @property
    def match_time_s(self) -> float:
        """Match time the candidate refers to."""
        return self.prediction.match_time_s


@dataclass(frozen=True, slots=True)
class Insight:
    """An insight approved for display."""

    kind: InsightKind
    headline: str
    detail: str
    probability: float
    match_time_s: float
    period: int
    attacking_team: str
    model_name: str
    model_version: str
    is_ml: bool
    facts: tuple[ContextFact, ...] = ()
    emitted_at_s: float = 0.0

    def to_dict(self) -> JsonDict:
        """Serialisable form sent over the stream."""
        return {
            "kind": self.kind.value,
            "headline": self.headline,
            "detail": self.detail,
            "probability": round(self.probability, 4),
            "match_time_s": round(self.match_time_s, 2),
            "period": self.period,
            "attacking_team": self.attacking_team,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_ml": self.is_ml,
            "facts": [{"key": f.key, "text": f.text, "value": f.value} for f in self.facts],
        }


@dataclass(frozen=True, slots=True)
class Suppressed:
    """A candidate the editorial layer declined to show."""

    reason: SuppressionReason
    match_time_s: float
    probability: float
    kind: InsightKind | None = None


@dataclass(frozen=True, slots=True)
class EditorialOutcome:
    """The result of editorial review: at most one insight, plus the reason if not."""

    insight: Insight | None = None
    suppressed: Suppressed | None = None
    #: Reasons evaluated before the decision, for debugging and reporting.
    trace: tuple[str, ...] = field(default_factory=tuple)

    @property
    def emitted(self) -> bool:
        """Whether an insight reached the viewer."""
        return self.insight is not None
