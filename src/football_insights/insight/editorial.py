"""Editorial selection: deciding what a viewer actually sees.

This layer sits between a scored window and the screen, and it is where most of
the product judgement lives. The model answers "how likely is a penalty-area
entry in the next few seconds"; this answers "is telling someone that right now
useful, or noise".

It is measured separately from the model. A run that emits three insights from
four hundred above-threshold windows has not made the model better or worse — it
has changed what the audience experiences, and that is reported on its own terms.

Check order matters, because it determines which reason gets recorded. Data
problems are evaluated before confidence so that a broken feed is never
misreported as a quiet model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.insight.templates import (
    build_detail,
    choose_kind,
    extract_facts,
    headline_for,
)
from football_insights.insight.types import (
    EditorialOutcome,
    Insight,
    InsightCandidate,
    InsightKind,
    Prediction,
    Suppressed,
    SuppressionReason,
)

if TYPE_CHECKING:
    from football_insights.config import EditorialSettings

#: Probability at or above which the stronger wording is used. Expressed as a
#: fraction of the way from the decision threshold to certainty, so it tracks a
#: retuned threshold instead of silently drifting out of step with it.
HIGH_BAND_FRACTION = 0.45


@dataclass(slots=True)
class _Emission:
    """A previously emitted insight, kept for cooldown and duplicate checks."""

    kind: InsightKind
    headline: str
    match_time_s: float


class EditorialPolicy:
    """Applies relevance and suppression rules to insight candidates.

    Args:
        settings: Editorial configuration.
        threshold: Probability at or above which a prediction becomes a candidate.
        spec: Feature schema used to read context out of the window.
    """

    __slots__ = ("_consecutive", "_history", "_last_by_kind", "_settings", "_spec", "_threshold")

    def __init__(
        self,
        settings: EditorialSettings,
        threshold: float,
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    ) -> None:
        """Create a policy with no emission history."""
        self._settings = settings
        self._threshold = threshold
        self._spec = spec
        self._history: deque[_Emission] = deque(maxlen=64)
        self._last_by_kind: dict[InsightKind, float] = {}
        self._consecutive = 0

    def reset(self) -> None:
        """Forget all state.

        Called when replay restarts or the stream is interrupted, so an insight
        from a previous passage of play can never appear against new frames.
        """
        self._history.clear()
        self._last_by_kind.clear()
        self._consecutive = 0

    @property
    def high_band(self) -> float:
        """Probability above which the stronger wording is used."""
        return self._threshold + (1.0 - self._threshold) * HIGH_BAND_FRACTION

    def build_candidate(
        self, prediction: Prediction, window: np.ndarray
    ) -> InsightCandidate | None:
        """Turn a prediction that cleared the threshold into a candidate.

        Args:
            prediction: The model output.
            window: The scored window, used for factual context.

        Returns:
            A candidate, or ``None`` if the prediction is below threshold.
        """
        if prediction.probability < self._threshold:
            return None
        facts = extract_facts(window, self._spec)
        last = np.asarray(window)[-1]
        kind = choose_kind(
            probability=prediction.probability,
            recent_entry_count=float(last[self._spec.index("recent_box_entry_count")]),
            possession_duration_s=float(last[self._spec.index("possession_duration_s")]),
            high_band=self.high_band,
        )
        return InsightCandidate(kind=kind, prediction=prediction, facts=facts)

    def review(
        self,
        prediction: Prediction,
        window: np.ndarray | None,
        now_s: float | None = None,
    ) -> EditorialOutcome:
        """Decide whether a scored window becomes a viewer-facing insight.

        Args:
            prediction: The model output for this window.
            window: The scored window, or ``None`` when it was invalid.
            now_s: Current match time. Defaults to the prediction's own time;
                a later value means the prediction arrived late.

        Returns:
            The outcome, carrying either an insight or a suppression reason.
        """
        now = prediction.match_time_s if now_s is None else now_s

        def suppress(
            reason: SuppressionReason, kind: InsightKind | None = None
        ) -> EditorialOutcome:
            self._consecutive = 0
            return EditorialOutcome(
                suppressed=Suppressed(
                    reason=reason,
                    match_time_s=prediction.match_time_s,
                    probability=prediction.probability,
                    kind=kind,
                )
            )

        # 1. Data integrity first: a broken feed must never be reported as a
        #    quiet model.
        if not prediction.window_valid or window is None:
            reason = prediction.invalid_reason or SuppressionReason.INVALID_WINDOW.value
            mapped = (
                SuppressionReason.INSUFFICIENT_FRAMES
                if reason == SuppressionReason.INSUFFICIENT_FRAMES.value
                else SuppressionReason.INVALID_WINDOW
            )
            return suppress(mapped)

        last = np.asarray(window)[-1]

        # 2. Nothing to say while the ball is out of play.
        if last[self._spec.index("is_dead_ball")] > 0.5:
            return suppress(SuppressionReason.DEAD_BALL)

        # 3. Nothing left to predict once the ball is already in the box.
        if self._settings.suppress_when_in_box and last[self._spec.index("ball_in_box")] > 0.5:
            return suppress(SuppressionReason.ALREADY_IN_BOX)

        # 4. A legitimate low-confidence prediction, distinct from a failure.
        candidate = self.build_candidate(prediction, window)
        if candidate is None:
            return suppress(SuppressionReason.LOW_CONFIDENCE)

        # 5. Require the signal to persist, damping single-window spikes.
        self._consecutive += 1
        if self._consecutive < self._settings.min_consecutive_windows:
            return EditorialOutcome(
                suppressed=Suppressed(
                    reason=SuppressionReason.NOT_YET_SUSTAINED,
                    match_time_s=prediction.match_time_s,
                    probability=prediction.probability,
                    kind=candidate.kind,
                )
            )

        # 6. Never show a situation that has already moved on.
        if now - prediction.match_time_s > self._settings.max_staleness_s:
            return suppress(SuppressionReason.STALE_SITUATION, candidate.kind)

        # 7. Respect the per-kind cooldown.
        last_emitted = self._last_by_kind.get(candidate.kind)
        if last_emitted is not None and now - last_emitted < self._settings.cooldown_s:
            return EditorialOutcome(
                suppressed=Suppressed(
                    reason=SuppressionReason.COOLDOWN,
                    match_time_s=prediction.match_time_s,
                    probability=prediction.probability,
                    kind=candidate.kind,
                )
            )

        headline = headline_for(candidate.kind)
        detail = build_detail(candidate.facts)

        # 8. Do not repeat wording a viewer has just read.
        for previous in reversed(self._history):
            if now - previous.match_time_s > self._settings.duplicate_window_s:
                break
            if previous.headline == headline:
                return EditorialOutcome(
                    suppressed=Suppressed(
                        reason=SuppressionReason.DUPLICATE_RECENT,
                        match_time_s=prediction.match_time_s,
                        probability=prediction.probability,
                        kind=candidate.kind,
                    )
                )

        insight = Insight(
            kind=candidate.kind,
            headline=headline,
            detail=detail,
            probability=prediction.probability,
            match_time_s=prediction.match_time_s,
            period=prediction.period,
            attacking_team=prediction.attacking_team,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            is_ml=prediction.is_ml,
            facts=candidate.facts,
            emitted_at_s=now,
        )
        self._last_by_kind[candidate.kind] = now
        self._history.append(_Emission(candidate.kind, headline, now))
        return EditorialOutcome(insight=insight)

    def suppress_unavailable(
        self, match_time_s: float, reason: SuppressionReason
    ) -> EditorialOutcome:
        """Record a service-level failure as a suppression.

        Used when no predictor is loaded or the feature schema does not match.
        Distinguishing this from a low-confidence prediction is what lets an
        operator tell "the model is quiet" from "the model is missing".

        Args:
            match_time_s: Current match time.
            reason: Either ``MODEL_UNAVAILABLE`` or ``SCHEMA_MISMATCH``.

        Returns:
            A suppressed outcome.
        """
        self._consecutive = 0
        return EditorialOutcome(
            suppressed=Suppressed(reason=reason, match_time_s=match_time_s, probability=0.0)
        )
