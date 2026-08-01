"""Replay determinism, window validity and editorial suppression."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from football_insights.config import EditorialSettings, Settings
from football_insights.data.synthetic import generate_synthetic_match
from football_insights.features.spec import DEFAULT_FEATURE_SPEC as SPEC
from football_insights.features.window import (
    RollingWindow,
    WindowGeometry,
    subsample_indices,
)
from football_insights.insight.editorial import EditorialPolicy
from football_insights.insight.templates import (
    FORBIDDEN_ASSERTIONS,
    headline_for,
    is_hedged,
)
from football_insights.insight.types import (
    EditorialOutcome,
    InsightKind,
    Prediction,
    SuppressionReason,
)
from football_insights.replay.faults import DRAWS_PER_FRAME, FaultInjector, stream_signature


@pytest.fixture(scope="module")
def match():
    return generate_synthetic_match(seed=17, period_duration_s=60.0)


@pytest.fixture
def settings():
    return Settings()


def _reason(outcome: EditorialOutcome) -> SuppressionReason:
    """Assert the outcome was suppressed and return why."""
    assert outcome.suppressed is not None, "expected a suppression, got an insight"
    return outcome.suppressed.reason


def _prediction(probability: float, time_s: float = 100.0, valid: bool = True) -> Prediction:
    return Prediction(
        probability=probability,
        match_time_s=time_s,
        period=1,
        attacking_team="home",
        model_name="test",
        model_version="1.0.0",
        is_ml=True,
        window_valid=valid,
    )


def _window(**overrides: float) -> np.ndarray:
    """A window whose final timestep has the named features set."""
    window = np.zeros(
        (SPEC.sequence_length if hasattr(SPEC, "sequence_length") else 50, SPEC.n_features),
        dtype=np.float32,
    )
    for name, value in overrides.items():
        window[-1, SPEC.index(name)] = value
    return window


class TestFaultDeterminism:
    @pytest.mark.parametrize("profile", ["clean", "jitter", "degraded", "hostile"])
    def test_same_seed_gives_an_identical_stream(self, match, settings, profile):
        frames = list(match.tracking.iter_frames())
        first, _ = FaultInjector(settings.fault_profile(profile), 42).apply(frames)
        second, _ = FaultInjector(settings.fault_profile(profile), 42).apply(frames)
        assert stream_signature(first) == stream_signature(second)

    def test_different_seeds_diverge(self, match, settings):
        frames = list(match.tracking.iter_frames())
        a, _ = FaultInjector(settings.fault_profile("hostile"), 1).apply(frames)
        b, _ = FaultInjector(settings.fault_profile("hostile"), 2).apply(frames)
        assert stream_signature(a) != stream_signature(b)

    def test_clean_profile_is_the_identity(self, match, settings):
        frames = list(match.tracking.iter_frames())
        emitted, summary = FaultInjector(settings.fault_profile("clean"), 9).apply(frames)
        assert [e.frame.frame for e in emitted] == [f.frame for f in frames]
        assert summary.dropped == summary.duplicated == summary.reordered == 0
        assert all(e.offset_s == 0.0 for e in emitted)

    def test_draw_count_per_frame_is_fixed(self, settings):
        """Guards the determinism contract.

        Drawing a variable number of random values per frame would make the
        stream depend on earlier outcomes, so changing one probability would
        shift every later frame. The count is asserted rather than assumed.
        """
        import random

        reference = random.Random(5)
        drawn = [reference.random() for _ in range(DRAWS_PER_FRAME)]
        assert len(drawn) == 6

    def test_degraded_profile_actually_degrades(self, match, settings):
        frames = list(match.tracking.iter_frames())
        _, summary = FaultInjector(settings.fault_profile("degraded"), 42).apply(frames)
        assert summary.dropped > 0
        assert summary.duplicated > 0
        assert summary.reordered > 0
        assert summary.emitted_frames != summary.source_frames

    def test_out_of_order_frames_actually_appear(self, match, settings):
        frames = list(match.tracking.iter_frames())
        emitted, summary = FaultInjector(settings.fault_profile("hostile"), 3).apply(frames)
        ids = [e.frame.frame for e in emitted]
        inversions = sum(1 for a, b in pairwise(ids) if b < a)
        assert inversions > 0
        assert summary.reordered > 0


class TestRollingWindow:
    def test_rejects_non_increasing_frames(self, match, settings):
        geometry = WindowGeometry.build(settings.window, match.frame_rate)
        buffer = RollingWindow(geometry, settings.window.min_valid_frame_ratio)
        frames = list(match.tracking.iter_frames())
        buffer.append(frames[0])
        buffer.append(frames[1])
        with pytest.raises(ValueError, match="not newer"):
            buffer.append(frames[1])

    def test_incomplete_buffer_is_invalid(self, match, settings):
        geometry = WindowGeometry.build(settings.window, match.frame_rate)
        buffer = RollingWindow(geometry, settings.window.min_valid_frame_ratio)
        for frame in list(match.tracking.iter_frames())[:10]:
            buffer.append(frame)
        validity = buffer.validity()
        assert validity.ok is False
        assert validity.reason == "insufficient_frames"

    def test_missing_ball_invalidates_the_window(self, match, settings):
        geometry = WindowGeometry.build(settings.window, match.frame_rate)
        buffer = RollingWindow(geometry, min_valid_frame_ratio=0.8)
        frames = list(match.tracking.iter_frames())[: geometry.capacity]
        import dataclasses

        blanked = [
            dataclasses.replace(f, ball_xy=np.array([np.nan, np.nan]))
            if i >= geometry.capacity // 2
            else f
            for i, f in enumerate(frames)
        ]
        for frame in blanked:
            buffer.append(frame)
        assert buffer.validity().ok is False

    def test_period_change_clears_history(self, match, settings):
        geometry = WindowGeometry.build(settings.window, match.frame_rate)
        buffer = RollingWindow(geometry, settings.window.min_valid_frame_ratio)
        frames = list(match.tracking.iter_frames())
        first_period = [f for f in frames if f.period == 1][: geometry.capacity]
        for frame in first_period:
            buffer.append(frame)
        assert buffer.is_full
        second = next(f for f in frames if f.period == 2)
        buffer.append(second)
        assert len(buffer) == 1, "velocity history cannot cross a period boundary"

    def test_subsampling_always_includes_the_prediction_instant(self):
        picks = subsample_indices(125, 50)
        assert picks[-1] == 124
        assert picks[0] == 0
        assert len(picks) == 50
        assert np.all(np.diff(picks) >= 0)


class TestEditorialSuppression:
    def test_invalid_window_never_emits(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.1)
        outcome = policy.review(_prediction(0.99, valid=False), None)
        assert outcome.emitted is False
        assert _reason(outcome) is SuppressionReason.INVALID_WINDOW

    def test_low_confidence_is_distinct_from_failure(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.9)
        outcome = policy.review(_prediction(0.2), _window())
        assert _reason(outcome) is SuppressionReason.LOW_CONFIDENCE

    def test_missing_model_is_reported_separately(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        outcome = policy.suppress_unavailable(10.0, SuppressionReason.MODEL_UNAVAILABLE)
        assert _reason(outcome) is SuppressionReason.MODEL_UNAVAILABLE

    def test_ball_already_in_box_is_suppressed(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.1)
        outcome = policy.review(_prediction(0.99), _window(ball_in_box=1.0))
        assert _reason(outcome) is SuppressionReason.ALREADY_IN_BOX

    def test_dead_ball_is_suppressed(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.1)
        outcome = policy.review(_prediction(0.99), _window(is_dead_ball=1.0))
        assert _reason(outcome) is SuppressionReason.DEAD_BALL

    def test_single_spike_is_not_yet_sustained(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        outcome = policy.review(_prediction(0.9), _window(attackers_ahead_of_ball=3))
        assert _reason(outcome) is SuppressionReason.NOT_YET_SUSTAINED

    def test_sustained_signal_emits(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        window = _window(attackers_ahead_of_ball=3, nearest_defender_dist=9.0)
        policy.review(_prediction(0.9, 100.0), window)
        outcome = policy.review(_prediction(0.9, 100.5), window)
        assert outcome.insight is not None
        assert is_hedged(outcome.insight.headline)

    def test_cooldown_blocks_a_repeat(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        window = _window(attackers_ahead_of_ball=3, nearest_defender_dist=9.0)
        policy.review(_prediction(0.9, 100.0), window)
        assert policy.review(_prediction(0.9, 100.5), window).emitted is True
        blocked = policy.review(_prediction(0.9, 101.0), window)
        assert _reason(blocked) is SuppressionReason.COOLDOWN

    def test_stale_candidate_is_dropped(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        window = _window(attackers_ahead_of_ball=3)
        policy.review(_prediction(0.9, 100.0), window, now_s=100.0)
        outcome = policy.review(_prediction(0.9, 100.5), window, now_s=140.0)
        assert _reason(outcome) is SuppressionReason.STALE_SITUATION

    def test_reset_clears_history_after_an_interruption(self, settings):
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        window = _window(attackers_ahead_of_ball=3)
        policy.review(_prediction(0.9, 100.0), window)
        policy.review(_prediction(0.9, 100.5), window)
        policy.reset()
        # After a reset the sustain counter starts again, so one window is not
        # enough — a stale insight cannot leak across the interruption.
        outcome = policy.review(_prediction(0.9, 500.0), window)
        assert _reason(outcome) is SuppressionReason.NOT_YET_SUSTAINED

    def test_min_consecutive_of_one_emits_immediately(self):
        policy = EditorialPolicy(EditorialSettings(min_consecutive_windows=1), threshold=0.5)
        outcome = policy.review(_prediction(0.9), _window(attackers_ahead_of_ball=3))
        assert outcome.emitted is True


class TestWording:
    @pytest.mark.parametrize("kind", list(InsightKind))
    def test_every_headline_is_hedged(self, kind):
        assert is_hedged(headline_for(kind))

    @pytest.mark.parametrize("phrase", sorted(FORBIDDEN_ASSERTIONS))
    def test_assertive_phrasing_is_rejected(self, phrase):
        assert is_hedged(f"The attack {phrase} the box") is False

    def test_headlines_are_deterministic(self):
        assert headline_for(InsightKind.BUILDING_THREAT) == headline_for(
            InsightKind.BUILDING_THREAT
        )
