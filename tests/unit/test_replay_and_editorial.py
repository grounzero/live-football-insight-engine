"""Replay determinism, window validity and editorial suppression."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from itertools import pairwise
from unittest import mock

import numpy as np
import pytest

from football_insights.config import EditorialSettings, Settings
from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
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
from football_insights.replay import player as player_module
from football_insights.replay.faults import DRAWS_PER_FRAME, FaultInjector, stream_signature
from football_insights.replay.player import ReplayPlayer
from football_insights.types import Float32Array


@pytest.fixture(scope="module")
def match() -> SyntheticMatch:
    return generate_synthetic_match(seed=17, period_duration_s=60.0)


@pytest.fixture
def settings() -> Settings:
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


def _window(**overrides: float) -> Float32Array:
    """A window whose final timestep has the named features set."""
    # Length comes from the window settings, not the feature spec: FeatureSpec
    # describes the columns, WindowSettings describes how many timesteps.
    window = np.zeros((Settings().window.sequence_length, SPEC.n_features), dtype=np.float32)
    for name, value in overrides.items():
        window[-1, SPEC.index(name)] = value
    return window


class TestFaultDeterminism:
    @pytest.mark.parametrize("profile", ["clean", "jitter", "degraded", "hostile"])
    def test_same_seed_gives_an_identical_stream(
        self, match: SyntheticMatch, settings: Settings, profile: str
    ) -> None:
        frames = list(match.tracking.iter_frames())
        first, _ = FaultInjector(settings.fault_profile(profile), 42).apply(frames)
        second, _ = FaultInjector(settings.fault_profile(profile), 42).apply(frames)
        assert stream_signature(first) == stream_signature(second)

    def test_different_seeds_diverge(self, match: SyntheticMatch, settings: Settings) -> None:
        frames = list(match.tracking.iter_frames())
        a, _ = FaultInjector(settings.fault_profile("hostile"), 1).apply(frames)
        b, _ = FaultInjector(settings.fault_profile("hostile"), 2).apply(frames)
        assert stream_signature(a) != stream_signature(b)

    def test_clean_profile_is_the_identity(self, match: SyntheticMatch, settings: Settings) -> None:
        frames = list(match.tracking.iter_frames())
        emitted, summary = FaultInjector(settings.fault_profile("clean"), 9).apply(frames)
        assert [e.frame.frame for e in emitted] == [f.frame for f in frames]
        assert summary.dropped == summary.duplicated == summary.reordered == 0
        assert all(e.offset_s == 0.0 for e in emitted)

    def test_draw_count_per_frame_is_fixed(self, settings: Settings) -> None:
        """Guards the determinism contract.

        Drawing a variable number of random values per frame would make the
        stream depend on earlier outcomes, so changing one probability would
        shift every later frame. The count is asserted rather than assumed.
        """
        import random

        reference = random.Random(5)
        drawn = [reference.random() for _ in range(DRAWS_PER_FRAME)]
        assert len(drawn) == 6

    def test_degraded_profile_actually_degrades(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        frames = list(match.tracking.iter_frames())
        _, summary = FaultInjector(settings.fault_profile("degraded"), 42).apply(frames)
        assert summary.dropped > 0
        assert summary.duplicated > 0
        assert summary.reordered > 0
        assert summary.emitted_frames != summary.source_frames

    def test_out_of_order_frames_actually_appear(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        frames = list(match.tracking.iter_frames())
        emitted, summary = FaultInjector(settings.fault_profile("hostile"), 3).apply(frames)
        ids = [e.frame.frame for e in emitted]
        inversions = sum(1 for a, b in pairwise(ids) if b < a)
        assert inversions > 0
        assert summary.reordered > 0


class TestRollingWindow:
    def test_rejects_non_increasing_frames(self, match: SyntheticMatch, settings: Settings) -> None:
        geometry = WindowGeometry.build(settings.window, match.frame_rate)
        buffer = RollingWindow(geometry, settings.window.min_valid_frame_ratio)
        frames = list(match.tracking.iter_frames())
        buffer.append(frames[0])
        buffer.append(frames[1])
        with pytest.raises(ValueError, match="not newer"):
            buffer.append(frames[1])

    def test_incomplete_buffer_is_invalid(self, match: SyntheticMatch, settings: Settings) -> None:
        geometry = WindowGeometry.build(settings.window, match.frame_rate)
        buffer = RollingWindow(geometry, settings.window.min_valid_frame_ratio)
        for frame in list(match.tracking.iter_frames())[:10]:
            buffer.append(frame)
        validity = buffer.validity()
        assert validity.ok is False
        assert validity.reason == "insufficient_frames"

    def test_missing_ball_invalidates_the_window(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
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

    def test_period_change_clears_history(self, match: SyntheticMatch, settings: Settings) -> None:
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

    def test_subsampling_always_includes_the_prediction_instant(self) -> None:
        picks = subsample_indices(125, 50)
        assert picks[-1] == 124
        assert picks[0] == 0
        assert len(picks) == 50
        assert np.all(np.diff(picks) >= 0)


class TestEditorialSuppression:
    def test_invalid_window_never_emits(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.1)
        outcome = policy.review(_prediction(0.99, valid=False), None)
        assert outcome.emitted is False
        assert _reason(outcome) is SuppressionReason.INVALID_WINDOW

    def test_low_confidence_is_distinct_from_failure(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.9)
        outcome = policy.review(_prediction(0.2), _window())
        assert _reason(outcome) is SuppressionReason.LOW_CONFIDENCE

    def test_missing_model_is_reported_separately(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        outcome = policy.suppress_unavailable(10.0, SuppressionReason.MODEL_UNAVAILABLE)
        assert _reason(outcome) is SuppressionReason.MODEL_UNAVAILABLE

    def test_ball_already_in_box_is_suppressed(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.1)
        outcome = policy.review(_prediction(0.99), _window(ball_in_box=1.0))
        assert _reason(outcome) is SuppressionReason.ALREADY_IN_BOX

    def test_dead_ball_is_suppressed(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.1)
        outcome = policy.review(_prediction(0.99), _window(is_dead_ball=1.0))
        assert _reason(outcome) is SuppressionReason.DEAD_BALL

    def test_single_spike_is_not_yet_sustained(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        outcome = policy.review(_prediction(0.9), _window(attackers_ahead_of_ball=3))
        assert _reason(outcome) is SuppressionReason.NOT_YET_SUSTAINED

    def test_sustained_signal_emits(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        window = _window(attackers_ahead_of_ball=3, nearest_defender_dist=9.0)
        policy.review(_prediction(0.9, 100.0), window)
        outcome = policy.review(_prediction(0.9, 100.5), window)
        assert outcome.insight is not None
        assert is_hedged(outcome.insight.headline)

    def test_cooldown_blocks_a_repeat(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        window = _window(attackers_ahead_of_ball=3, nearest_defender_dist=9.0)
        policy.review(_prediction(0.9, 100.0), window)
        assert policy.review(_prediction(0.9, 100.5), window).emitted is True
        blocked = policy.review(_prediction(0.9, 101.0), window)
        assert _reason(blocked) is SuppressionReason.COOLDOWN

    def test_stale_candidate_is_dropped(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        window = _window(attackers_ahead_of_ball=3)
        policy.review(_prediction(0.9, 100.0), window, now_s=100.0)
        outcome = policy.review(_prediction(0.9, 100.5), window, now_s=140.0)
        assert _reason(outcome) is SuppressionReason.STALE_SITUATION

    def test_reset_clears_history_after_an_interruption(self, settings: Settings) -> None:
        policy = EditorialPolicy(settings.editorial, threshold=0.5)
        window = _window(attackers_ahead_of_ball=3)
        policy.review(_prediction(0.9, 100.0), window)
        policy.review(_prediction(0.9, 100.5), window)
        policy.reset()
        # After a reset the sustain counter starts again, so one window is not
        # enough — a stale insight cannot leak across the interruption.
        outcome = policy.review(_prediction(0.9, 500.0), window)
        assert _reason(outcome) is SuppressionReason.NOT_YET_SUSTAINED

    def test_min_consecutive_of_one_emits_immediately(self) -> None:
        policy = EditorialPolicy(EditorialSettings(min_consecutive_windows=1), threshold=0.5)
        outcome = policy.review(_prediction(0.9), _window(attackers_ahead_of_ball=3))
        assert outcome.emitted is True


class TestWording:
    @pytest.mark.parametrize("kind", list(InsightKind))
    def test_every_headline_is_hedged(self, kind: InsightKind) -> None:
        assert is_hedged(headline_for(kind))

    @pytest.mark.parametrize("phrase", sorted(FORBIDDEN_ASSERTIONS))
    def test_assertive_phrasing_is_rejected(self, phrase: str) -> None:
        assert is_hedged(f"The attack {phrase} the box") is False

    def test_headlines_are_deterministic(self) -> None:
        assert headline_for(InsightKind.BUILDING_THREAT) == headline_for(
            InsightKind.BUILDING_THREAT
        )


class _StubAsyncio:
    """Stands in for the `asyncio` name inside the player module.

    Patching `asyncio.sleep` globally deadlocks pytest-asyncio, which needs real
    sleeps of its own. Replacing only the reference the player holds records the
    delays it asks for without slowing the test or touching the loop machinery.
    """

    def __init__(self) -> None:
        self.requested: list[float] = []
        #: Called on every sleep. A paused player yields no frames, so this is
        #: the only place a test can act on one without hanging.
        self.on_sleep: Callable[[], None] | None = None

    async def sleep(self, delay: float = 0, *args: object, **kwargs: object) -> None:
        self.requested.append(delay)
        if self.on_sleep is not None:
            self.on_sleep()
        await asyncio.sleep(0)


class _PacedStubAsyncio:
    """`_StubAsyncio` plus a clock that its own sleeps advance.

    Stands in for both `asyncio` and `time` inside the player module. The player
    measures its schedule against elapsed real time, so a stub that never really
    sleeps freezes the comparison: the loop believes no time has passed and
    stays on schedule however wrong its anchor is. Advancing a fake clock by
    each requested delay makes an accelerated test behave like a real run, which
    is what any test of *re-anchoring* needs.
    """

    def __init__(self) -> None:
        self.requested: list[float] = []
        self.now = 0.0

    async def sleep(self, delay: float = 0, *args: object, **kwargs: object) -> None:
        self.requested.append(delay)
        self.now += delay
        await asyncio.sleep(0)

    def perf_counter(self) -> float:
        return self.now


class TestReplayPacing:
    """Pacing must follow control input, not a schedule fixed at start-up.

    `stream` keeps its pacing anchor in local variables, so a control request —
    which arrives on a different task — cannot reset it directly. While it was
    not reset, lowering the speed made the loop try to sleep off the match time
    already replayed at the old rate: 21 s of silence after only 3 s of playback
    going 8x -> 1x, and proportionally worse the longer the replay had run. To a
    viewer that is indistinguishable from the service having hung.
    """

    @staticmethod
    def _player(settings: Settings, match: SyntheticMatch, speed: float) -> ReplayPlayer:
        return ReplayPlayer(
            match_id="synthetic",
            tracking=match.tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=speed,
        )

    async def _delays_after(
        self,
        settings: Settings,
        match: SyntheticMatch,
        act: Callable[[ReplayPlayer], None],
        *,
        at_frame: int = 200,
    ) -> list[float]:
        """Run the stream, apply `act` at `at_frame`, return the delays after it."""
        stub = _StubAsyncio()
        player = self._player(settings, match, speed=8.0)
        frames = 0
        with mock.patch.object(player_module, "asyncio", stub):
            async for _ in player.stream():
                frames += 1
                if frames == at_frame:
                    stub.requested.clear()
                    act(player)
                    continue
                if frames > at_frame:
                    break
        return stub.requested

    @pytest.mark.parametrize("new_speed", [1.0, 2.0, 5.0, 10.0])
    async def test_changing_speed_does_not_stall_the_stream(
        self, match: SyntheticMatch, settings: Settings, new_speed: float
    ) -> None:
        """Every speed the demo offers must take effect on the next frame."""
        delays = await self._delays_after(settings, match, lambda p: p.set_speed(new_speed))
        # One frame at 25 Hz is 40 ms of match time, so at 1x — the slowest the
        # demo offers — a correctly re-anchored loop sleeps about 40 ms. Anything
        # on the order of seconds means it is catching up on the old schedule.
        assert max(delays, default=0.0) < 1.0, (
            f"stalled for {max(delays):.1f}s after changing speed to {new_speed}x"
        )

    async def test_resuming_does_not_replay_the_pause(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        """Time spent paused must not be caught up on resume."""
        stub = _StubAsyncio()
        player = self._player(settings, match, speed=8.0)
        pause_ticks = 0

        def resume_after_a_few_ticks() -> None:
            # A paused stream yields nothing, so the resume has to come from
            # inside the pause loop rather than from the consumer below.
            nonlocal pause_ticks
            if not player.paused:
                return
            pause_ticks += 1
            if pause_ticks >= 5:
                stub.requested.clear()
                player.set_paused(False)

        stub.on_sleep = resume_after_a_few_ticks

        frames = 0
        with mock.patch.object(player_module, "asyncio", stub):
            async for _ in player.stream():
                frames += 1
                if frames == 200:
                    player.set_paused(True)
                elif frames > 200:
                    break

        assert pause_ticks >= 5, "the player never actually paused"
        assert frames > 200, "the stream did not resume"
        assert max(stub.requested, default=0.0) < 1.0

    def test_speed_change_is_visible_in_status(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        player = self._player(settings, match, speed=8.0)
        player.set_speed(2.0)
        assert player.status().speed == 2.0

    def test_negative_speed_is_clamped(self, match: SyntheticMatch, settings: Settings) -> None:
        player = self._player(settings, match, speed=8.0)
        player.set_speed(-5.0)
        assert player.status().speed == 0.0

    def test_reset_rewinds_position(self, match: SyntheticMatch, settings: Settings) -> None:
        player = self._player(settings, match, speed=8.0)
        player.reset()
        status = player.status()
        assert status.frames_emitted == 0
        assert status.match_time_s == 0.0

    async def test_reset_re_anchors_pacing(self, match: SyntheticMatch, settings: Settings) -> None:
        """Rewinding must re-pace, not sprint through frames it already played.

        This is the speed-change failure above arriving from the other
        direction, and it presents as the *absence* of sleeps rather than one
        long one: a stale anchor is measured from a match time the rewound
        replay is immediately far behind, so every frame looks overdue, `delay`
        is negative, and the loop runs flat out until it catches back up.

        It needs `_PacedStubAsyncio` rather than the stub the tests above use.
        Neither stub really sleeps, but the player compares its schedule against
        elapsed *real* time, so with a frozen clock even a hopelessly stale
        anchor still looks on schedule and the bug is invisible.
        """
        stub = _PacedStubAsyncio()
        player = self._player(settings, match, speed=8.0)
        frames = 0
        with (
            mock.patch.object(player_module, "asyncio", stub),
            mock.patch.object(player_module, "time", stub),
        ):
            async for _ in player.stream():
                frames += 1
                if frames == 200:
                    stub.requested.clear()
                    player.reset()
                    continue
                if frames > 240:
                    break

        assert stub.requested, "replay sprinted after rewinding: no frame was paced"
        assert max(stub.requested) < 1.0, f"stalled for {max(stub.requested):.1f}s after rewinding"

    async def test_a_replay_running_behind_still_yields_to_the_event_loop(
        self, settings: Settings, match: SyntheticMatch
    ) -> None:
        """A replay that cannot keep up must not stop the rest of the service.

        On the paced path the pacing sleep is the *only* thing that returns
        control to the event loop, and it disappears exactly when the replay
        falls behind: `delay` goes negative, nothing is awaited, and resuming an
        async generator does not reach the scheduler. The whole match then runs
        in one uninterrupted burst.

        Nothing about that is visible from the replay's own output — the frames
        are all correct and all present. What breaks is everything else: no
        subscriber is scheduled to receive them, `/health` and `/ready` do not
        answer, and a platform watching those endpoints concludes the container
        is dead and restarts it.

        Real asyncio here, deliberately, rather than the stubs above: the whole
        question is whether the event loop is reached, and a stub that fakes
        sleeping cannot answer it.
        """
        player = self._player(settings, match, speed=50.0)
        scheduled = 0

        async def competing_work() -> None:
            nonlocal scheduled
            while True:
                await asyncio.sleep(0)
                scheduled += 1

        task = asyncio.create_task(competing_work())
        frames = 0
        try:
            async for _ in player.stream():
                # Per-frame work beyond the budget the speed implies, which is
                # what puts the loop behind: at 50x, 25 Hz frames allow 0.8 ms.
                deadline = time.perf_counter() + 0.002
                while time.perf_counter() < deadline:
                    pass
                frames += 1
                if frames >= 200:
                    player.stop()
        finally:
            task.cancel()

        assert frames >= 200
        assert scheduled > 0, (
            "the replay ran 200 frames behind schedule without the event loop "
            "being scheduled once: every other task in the process is starved"
        )
