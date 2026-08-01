"""The live inference engine.

Consumes frames one at a time, maintains the rolling window and causal
possession state incrementally, scores each window and passes the result
through editorial review. The full match is never re-read: appending a frame is
constant work regardless of how long the replay has been running.

The engine owns the boundary between "the model is quiet" and "the system is
broken". A missing predictor, a schema mismatch or an unusable window all
produce a suppression with a specific reason rather than silence, so an
operator can tell the cases apart from the metrics alone.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from football_insights.domain import EventType, Team
from football_insights.errors import SchemaVersionError
from football_insights.features.causal import CausalEventView
from football_insights.features.frame_features import PossessionContext
from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.features.window import (
    RollingWindow,
    WindowGeometry,
    window_features_from_buffer,
)
from football_insights.insight.editorial import EditorialPolicy
from football_insights.insight.types import (
    EditorialOutcome,
    Prediction,
    SuppressionReason,
)
from football_insights.pitch import DEFAULT_PITCH, Pitch

if TYPE_CHECKING:
    from football_insights.config import Settings
    from football_insights.domain import Frame, Orientation
    from football_insights.models.base import Predictor
    from football_insights.serving.metrics import Metrics

#: Lookback for the recent penalty-area entry context feature.
BOX_ENTRY_LOOKBACK_S = 120.0
#: Lookback for the recent pass count.
RECENT_PASS_LOOKBACK_S = 10.0


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Outcome of processing one frame."""

    prediction: Prediction | None
    outcome: EditorialOutcome | None
    frame_accepted: bool
    #: Set when the frame could not be processed at all, e.g. it arrived out of
    #: order. Distinct from a suppression, which implies a scored window.
    rejection: str | None = None


class _ContextTracker:
    """Maintains causal possession context incrementally.

    Recomputing possession over the whole buffer on every frame would be
    wasteful and, worse, would tempt an implementation to look at the full event
    list. Here each frame contributes exactly one entry, derived from the causal
    view at that instant and never revised afterwards.
    """

    __slots__ = (
        "_counts",
        "_dead",
        "_durations",
        "_entry_counts",
        "_entry_times",
        "_flight",
        "_in_box_prev",
        "_passes",
        "_since_entry",
        "_view",
    )

    def __init__(self, view: CausalEventView, capacity: int) -> None:
        """Wire the buffer, context tracker, policy and predictor together."""
        self._view = view
        self._durations: deque[float] = deque(maxlen=capacity)
        self._counts: deque[float] = deque(maxlen=capacity)
        self._dead: deque[float] = deque(maxlen=capacity)
        self._flight: deque[float] = deque(maxlen=capacity)
        self._passes: deque[float] = deque(maxlen=capacity)
        self._entry_counts: deque[float] = deque(maxlen=capacity)
        self._since_entry: deque[float] = deque(maxlen=capacity)
        self._entry_times: list[float] = []
        self._in_box_prev = False

    def reset(self) -> None:
        """Drop all state after an interruption or period change."""
        for d in (
            self._durations,
            self._counts,
            self._dead,
            self._flight,
            self._passes,
            self._entry_counts,
            self._since_entry,
        ):
            d.clear()
        self._entry_times.clear()
        self._in_box_prev = False

    def push(self, frame: Frame, in_box: bool, frame_rate: float) -> Team | None:
        """Record context for one frame and return the team in possession."""
        state = self._view.possession(frame.frame)
        self._durations.append(state.duration_s)
        self._counts.append(float(state.event_count))
        self._dead.append(float(state.is_dead_ball))
        self._flight.append(float(state.has_event_in_flight))
        counts = self._view.recent_type_counts(
            frame.frame, int(RECENT_PASS_LOOKBACK_S * frame_rate)
        )
        self._passes.append(float(counts.get(EventType.PASS, 0) + counts.get(EventType.CARRY, 0)))

        # Drop entries that have fallen out of the lookback, then record this
        # frame's view of history *before* applying its own transition. A frame
        # must never count the entry it is currently making: that is the event
        # being predicted.
        self._entry_times = [
            t for t in self._entry_times if frame.time_s - t <= BOX_ENTRY_LOOKBACK_S
        ]
        self._entry_counts.append(float(len(self._entry_times)))
        self._since_entry.append(
            min(frame.time_s - self._entry_times[-1], BOX_ENTRY_LOOKBACK_S)
            if self._entry_times
            else BOX_ENTRY_LOOKBACK_S
        )
        if in_box and not self._in_box_prev:
            self._entry_times.append(frame.time_s)
        self._in_box_prev = in_box
        return state.team

    def context(self) -> PossessionContext:
        """Materialise the buffered context as arrays.

        Each value was recorded when its frame arrived and is never revised, so
        the per-frame variation matches what
        :func:`~football_insights.features.frame_features.box_entry_history`
        produces offline. Collapsing these to a single current value would be a
        subtle train/serve skew.
        """
        n = len(self._durations)
        return PossessionContext(
            duration_s=np.fromiter(self._durations, dtype=np.float64, count=n),
            event_count=np.fromiter(self._counts, dtype=np.float64, count=n),
            is_dead_ball=np.fromiter(self._dead, dtype=np.float64, count=n),
            event_in_flight=np.fromiter(self._flight, dtype=np.float64, count=n),
            recent_pass_count=np.fromiter(self._passes, dtype=np.float64, count=n),
            recent_box_entry_count=np.fromiter(self._entry_counts, dtype=np.float64, count=n),
            time_since_last_box_entry=np.fromiter(self._since_entry, dtype=np.float64, count=n),
        )


class InsightEngine:
    """Frame-at-a-time inference and editorial review.

    Args:
        settings: Resolved configuration.
        orientation: Attacking direction per period and team.
        events: Match events, wrapped in a causal view internally.
        frame_rate: Tracking sample rate in hertz.
        predictor: The scorer, or ``None`` to run without a model. With no
            predictor the engine stays functional and reports
            ``MODEL_UNAVAILABLE``; readiness is false and nothing is emitted.
        metrics: Metric sink.
        home_is_gk / away_is_gk: Goalkeeper column masks.
        spec: Feature schema.
        pitch: Pitch dimensions.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        orientation: Orientation,
        events: object,
        frame_rate: float,
        predictor: Predictor | None,
        metrics: Metrics,
        home_is_gk: np.ndarray | None = None,
        away_is_gk: np.ndarray | None = None,
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
        pitch: Pitch = DEFAULT_PITCH,
    ) -> None:
        """Wire the buffer, context tracker, policy and predictor together."""
        self._settings = settings
        self._orientation = orientation
        self._frame_rate = frame_rate
        self._metrics = metrics
        self._spec = spec
        self._pitch = pitch
        self._home_is_gk = home_is_gk
        self._away_is_gk = away_is_gk

        self._view = (
            events if isinstance(events, CausalEventView) else CausalEventView(events, frame_rate)  # type: ignore[arg-type]
        )
        self._geometry = WindowGeometry.build(settings.window, frame_rate)
        self._buffer = RollingWindow(
            self._geometry, settings.window.min_valid_frame_ratio, spec, pitch
        )
        self._tracker = _ContextTracker(self._view, self._geometry.capacity)
        self._policy = EditorialPolicy(settings.editorial, settings.model.threshold, spec)
        self._predictor = predictor
        self._schema_ok = True
        self._last_frame_id: int | None = None
        self._last_period: int | None = None

        if predictor is not None:
            try:
                predictor.metadata.require_schema(spec.schema_hash)
            except SchemaVersionError:
                self._schema_ok = False

    @property
    def geometry(self) -> WindowGeometry:
        """Window geometry in use."""
        return self._geometry

    @property
    def predictor(self) -> Predictor | None:
        """The active predictor, if any."""
        return self._predictor

    @property
    def is_ready(self) -> bool:
        """Whether the engine can produce predictions."""
        return self._predictor is not None and self._schema_ok

    @property
    def policy(self) -> EditorialPolicy:
        """The editorial policy, exposed for offline evaluation."""
        return self._policy

    def reset(self) -> None:
        """Clear all rolling state, e.g. when replay restarts."""
        self._buffer.clear()
        self._tracker.reset()
        self._policy.reset()
        self._last_frame_id = None
        self._last_period = None

    def process(self, frame: Frame) -> EngineResult:
        """Ingest one frame and produce at most one editorial outcome.

        Args:
            frame: The frame to process.

        Returns:
            The result, including whether the frame was accepted at all.
        """
        started = time.perf_counter()

        # Out-of-order and duplicate frames are rejected here. They are a
        # transport problem, not a model problem, and are counted separately.
        if self._last_frame_id is not None and frame.frame <= self._last_frame_id:
            self._metrics.replay_frames.labels(outcome="out_of_order").inc()
            return EngineResult(None, None, frame_accepted=False, rejection="out_of_order")
        if self._last_frame_id is not None and frame.frame > self._last_frame_id + 1:
            self._metrics.missing_frames.inc(frame.frame - self._last_frame_id - 1)

        try:
            self._buffer.append(frame)
        except ValueError:
            self._metrics.replay_frames.labels(outcome="rejected").inc()
            return EngineResult(None, None, frame_accepted=False, rejection="rejected")

        self._last_frame_id = frame.frame
        self._metrics.frames.inc()
        self._metrics.replay_frames.labels(outcome="accepted").inc()

        # The buffer clears itself at a period change; the context tracker must
        # follow, or the two lengths diverge and feature assembly fails.
        if self._last_period is not None and frame.period != self._last_period:
            self._tracker.reset()
        self._last_period = frame.period

        # Possession drives orientation, so it must be resolved before features.
        # The tracker is fed for *every* buffered frame, including ones where
        # possession is unknown: skipping those would leave the context arrays
        # shorter than the buffer they describe.
        attacking = self._provisional_attacker(frame)
        if attacking is None:
            self._tracker.push(frame, in_box=False, frame_rate=self._frame_rate)
            return EngineResult(None, None, frame_accepted=True)
        direction = self._orientation.direction(frame.period, attacking).sign
        oriented_ball = frame.ball_xy * direction
        in_box = bool(self._pitch.is_inside_penalty_area(oriented_ball))
        self._tracker.push(frame, in_box, self._frame_rate)

        validity = self._buffer.validity()
        if not validity.ok:
            self._metrics.invalid_windows.labels(reason=validity.reason).inc()
            prediction = Prediction(
                probability=0.0,
                match_time_s=frame.time_s,
                period=frame.period,
                attacking_team=attacking.value,
                model_name=self._model_name(),
                model_version=self._model_version(),
                is_ml=self._is_ml(),
                window_valid=False,
                invalid_reason=validity.reason,
            )
            outcome = self._policy.review(prediction, None, frame.time_s)
            self._record(outcome)
            self._metrics.e2e_latency.observe(time.perf_counter() - started)
            return EngineResult(prediction, outcome, frame_accepted=True)

        if self._predictor is None or not self._schema_ok:
            reason = (
                SuppressionReason.MODEL_UNAVAILABLE
                if self._predictor is None
                else SuppressionReason.SCHEMA_MISMATCH
            )
            outcome = self._policy.suppress_unavailable(frame.time_s, reason)
            self._record(outcome)
            self._metrics.e2e_latency.observe(time.perf_counter() - started)
            return EngineResult(None, outcome, frame_accepted=True)

        window = window_features_from_buffer(
            self._buffer,
            direction_sign=direction,
            frame_rate=self._frame_rate,
            possession=self._tracker.context(),
            attack_is_home=attacking is Team.HOME,
            attack_is_gk=self._home_is_gk if attacking is Team.HOME else self._away_is_gk,
            defend_is_gk=self._away_is_gk if attacking is Team.HOME else self._home_is_gk,
            spec=self._spec,
            pitch=self._pitch,
        )

        infer_started = time.perf_counter()
        probability = float(self._predictor.predict_proba(window[None, ...])[0])
        inference_s = time.perf_counter() - infer_started

        meta = self._predictor.metadata
        self._metrics.predictions.labels(model=meta.name, is_ml=str(meta.is_ml).lower()).inc()
        self._metrics.inference_latency.labels(
            model=meta.name, backend=self._settings.model.backend
        ).observe(inference_s)
        self._metrics.confidence.labels(model=meta.name).observe(probability)

        prediction = Prediction(
            probability=probability,
            match_time_s=frame.time_s,
            period=frame.period,
            attacking_team=attacking.value,
            model_name=meta.name,
            model_version=meta.version,
            is_ml=meta.is_ml,
            window_valid=True,
            inference_ms=inference_s * 1000.0,
        )
        outcome = self._policy.review(prediction, window, frame.time_s)
        self._record(outcome, probability)
        self._metrics.e2e_latency.observe(time.perf_counter() - started)
        return EngineResult(prediction, outcome, frame_accepted=True)

    def _provisional_attacker(self, frame: Frame) -> Team | None:
        """Team in possession at this frame, from the causal view only."""
        return self._view.possession(frame.frame).team

    def _record(self, outcome: EditorialOutcome, probability: float | None = None) -> None:
        """Update editorial metrics from a decision."""
        if outcome.insight is not None:
            self._metrics.candidates.labels(kind=outcome.insight.kind.value).inc()
            self._metrics.emitted.labels(
                kind=outcome.insight.kind.value, is_ml=str(outcome.insight.is_ml).lower()
            ).inc()
            return
        if outcome.suppressed is not None:
            suppressed = outcome.suppressed
            if suppressed.kind is not None:
                self._metrics.candidates.labels(kind=suppressed.kind.value).inc()
            self._metrics.suppressed.labels(reason=suppressed.reason.value).inc()
        _ = probability

    def _model_name(self) -> str:
        return self._predictor.metadata.name if self._predictor else "none"

    def _model_version(self) -> str:
        return self._predictor.metadata.version if self._predictor else "0"

    def _is_ml(self) -> bool:
        return bool(self._predictor and self._predictor.metadata.is_ml)
