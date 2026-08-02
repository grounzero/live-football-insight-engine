"""What reaches a browser, how often, and what must never be dropped.

The subject here is the split between *scoring* a frame and *showing* one. Every
canonical frame is scored; only about twenty a second are drawn. Getting that
wrong is not a visible bug on a development machine — the frames are all correct
and all present, and a local network delivers them evenly enough to hide it. It
shows up as stutter over the public internet, on someone else's connection,
where nobody is looking at a test suite.

So the assertions are about rate and losslessness rather than about pixels:
that the wire cadence is the same number at 1x and 8x, that a stalled producer
does not repay its missed slots in a burst, that insights and barriers cannot be
dropped to save bandwidth, and that a visitor arriving mid-replay is handed a
consistent picture rather than a blank pitch.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from football_insights.config import Settings
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.replay.player import ReplayPlayer
from football_insights.serving.loader import (
    SYNTHETIC_MATCH_ID,
    SyntheticDemoMatchSource,
    build_engine,
    demo_fixture_cycle,
)
from football_insights.serving.messages import StreamMessage, StreamMessageType, stream_message
from football_insights.serving.metrics import Metrics
from football_insights.serving.state import AppState, FixtureRotation
from football_insights.serving.stream import (
    VISUAL_PUBLISH_HZ,
    VisualRateLimiter,
    rotate_fixture,
    run_replay,
)
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeClock:
    """A monotonic clock a test advances by hand.

    The whole point of the limiter is that it measures wall time rather than
    frames, so a test of it that ran in real time would be measuring the machine
    it happens to be on. CI runners stall for tens of milliseconds without
    warning, which is exactly the length being asserted about.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FrameClock:
    """A wall clock that advances one frame's worth per reading.

    Lets a cadence test simulate any replay speed without running at it. The
    player is left unpaced, so the whole fixture streams through in
    milliseconds, while the limiter is shown the clock a perfectly paced replay
    at ``speed`` would have seen — one source frame every ``1 / (25 * speed)``
    seconds. Running a 1x test honestly would mean sleeping for the length of
    the fixture, and the thing being measured is a ratio, not a duration.

    The replay loop reads the clock exactly once per frame, inside the limiter,
    which is what makes one call the right unit to advance on.
    """

    def __init__(self, speed: float, source_hz: float = 25.0) -> None:
        self.now = 0.0
        self._step = 1.0 / (source_hz * speed)

    def __call__(self) -> float:
        now = self.now
        self.now += self._step
        return now

    @property
    def elapsed(self) -> float:
        """Simulated wall seconds consumed so far."""
        return self.now


def _counter(metrics: Metrics, name: str) -> float:
    """Read one counter out of the Prometheus exposition.

    Through ``render`` rather than the collector's internals, because that is
    the number a scraper actually receives — and because reaching into a
    counter's private storage would tie this test to the client library's
    implementation for no gain.

    Args:
        metrics: The registry to read.
        name: Full metric name, including the ``_total`` suffix.

    Returns:
        The counter's current value.
    """
    payload, _ = metrics.render()
    for line in payload.decode().splitlines():
        if line.startswith(f"{name} "):
            return float(line.split(" ", 1)[1])
    pytest.fail(f"{name} is not in the exposition")


class TestVisualRateLimiter:
    """The schedule itself, isolated from the replay it paces."""

    def test_the_first_frame_is_always_published(self) -> None:
        clock = _FakeClock()
        limiter = VisualRateLimiter(hz=20.0, clock=clock)

        # A viewer must not wait out a period before seeing anything.
        assert limiter.due() is True

    def test_frames_between_deadlines_are_withheld(self) -> None:
        clock = _FakeClock()
        limiter = VisualRateLimiter(hz=20.0, clock=clock)
        limiter.due()

        clock.advance(0.01)
        assert limiter.due() is False
        clock.advance(0.02)
        assert limiter.due() is False
        clock.advance(0.021)
        assert limiter.due() is True

    def test_a_missed_deadline_is_skipped_rather_than_repaid(self) -> None:
        """The behaviour that stops a stalled producer emitting a burst.

        Six slots pass while nothing is offered. Publishing six frames when the
        producer wakes puts five stale pictures on the wire in front of the only
        one still worth drawing, and the browser shows the whole backlog as a
        single snap.
        """
        clock = _FakeClock()
        limiter = VisualRateLimiter(hz=20.0, clock=clock)
        limiter.due()

        clock.advance(0.3)
        assert limiter.due() is True
        assert limiter.skipped_deadlines == 5

        # And the next frame, offered immediately afterwards, is not owed one.
        assert limiter.due() is False

    def test_forcing_publishes_and_re_anchors_the_schedule(self) -> None:
        clock = _FakeClock()
        limiter = VisualRateLimiter(hz=20.0, clock=clock)
        limiter.due()

        clock.advance(0.01)
        assert limiter.due(force=True) is True
        # Re-anchored, so the slot the old deadline named is not also spent:
        # a barrier must not cost the viewer the next scheduled picture.
        clock.advance(0.03)
        assert limiter.due() is False
        clock.advance(0.021)
        assert limiter.due() is True

    def test_reset_publishes_the_very_next_frame(self) -> None:
        clock = _FakeClock()
        limiter = VisualRateLimiter(hz=20.0, clock=clock)
        limiter.due()
        clock.advance(0.01)

        limiter.reset()
        # After a rewind the client has cleared its buffer and has nothing to
        # draw, so the next frame goes out whatever the schedule said.
        assert limiter.due() is True


def _state(settings: Settings, *, speed: float, period_s: float = 8.0) -> AppState:
    """A wired-up replay of a short generated fixture."""
    tracking, events, orientation = SyntheticDemoMatchSource(
        seed=3, n_periods=1, period_duration_s=period_s
    ).load()
    metrics = Metrics()
    engine = build_engine(
        settings,
        tracking,
        events,
        orientation,
        HeuristicPredictor(settings.model.threshold),
        metrics,
    )
    player = ReplayPlayer(
        match_id=SYNTHETIC_MATCH_ID,
        tracking=tracking,
        profile=settings.fault_profile("clean"),
        seed=1,
        speed=speed,
    )
    return AppState(settings=settings, metrics=metrics, engine=engine, player=player)


def _messages(queue: asyncio.Queue[StreamMessage]) -> Iterator[JsonDict]:
    """Every message currently queued, decoded."""
    while True:
        try:
            message = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        yield json.loads(message.data)


class TestPublicationCadence:
    """The wire rate must be a property of the wall clock, not of the replay."""

    @pytest.mark.parametrize("speed", [1.0, 4.0, 8.0])
    async def test_the_wire_rate_is_the_same_at_every_speed(self, speed: float) -> None:
        """The defect this whole change exists to fix.

        Publication used to be `counter % publish_every`, a ratio of source
        frames. Multiplied by the replay speed that came out at 25 messages a
        second at 1x and a hundred at 8x — a rate no display can show and no
        phone should be asked to parse. The number below must not move when the
        speed does.
        """
        settings = Settings()
        # Unpaced player, simulated clock: the cadence being asserted is a
        # ratio, so there is nothing to learn from waiting out the fixture in
        # real time at 1x.
        state = _state(settings, speed=0.0)
        queue = state.subscribe()
        clock = _FrameClock(speed)

        await asyncio.wait_for(run_replay(state, clock=clock), timeout=60.0)

        frames = [m for m in _messages(queue) if m["type"] == "frame"]
        elapsed = clock.elapsed
        assert elapsed > 0.5, "the fixture was too short to measure a rate"
        rate = len(frames) / elapsed

        assert 15.0 <= rate <= 25.0, (
            f"at {speed}x the wire carried {rate:.1f} visual frames a second; "
            f"the target is {VISUAL_PUBLISH_HZ}"
        )

    async def test_every_canonical_frame_is_still_scored(self) -> None:
        """Decimation is for display only; inference must not see a sample."""
        settings = Settings()
        state = _state(settings, speed=0.0)
        queue = state.subscribe()
        assert state.player is not None and state.engine is not None

        await asyncio.wait_for(run_replay(state, clock=_FrameClock(8.0)), timeout=60.0)

        published = [m for m in _messages(queue) if m["type"] == "frame"]
        scored = _counter(state.metrics, "fi_pipeline_frames_total")

        assert scored == state.player.total_frames, (
            "the engine did not process every canonical frame"
        )
        assert len(published) < scored / 2, (
            "the browser received nearly every frame; the visual limiter is not limiting"
        )

    async def test_no_burst_of_stale_frames_after_a_stall(self) -> None:
        """A producer that falls behind must not repay its missed slots at once.

        This is the failure mode a viewer sees as a snap: a long pause and then
        several frames' worth of movement applied faster than the display can
        paint it. The limiter's answer is to drop the intervening pictures, so
        the wire shows one frame per elapsed slot at most, never a backlog.
        """
        clock = _FakeClock()
        limiter = VisualRateLimiter(hz=20.0, clock=clock)
        limiter.due()

        # A 500 ms stall: the producer wakes with ten slots overdue and a queue
        # of frames it took while nobody was watching the clock.
        clock.advance(0.5)
        published = sum(1 for _ in range(50) if limiter.due())

        assert published == 1, f"{published} frames went out for one elapsed deadline"


class TestBarriersAndLosslessness:
    """What may never be dropped, whatever the bandwidth."""

    def test_a_full_queue_drops_pictures_but_never_barriers(self) -> None:
        settings = Settings()
        state = _state(settings, speed=0.0)
        queue = state.subscribe()

        for i in range(300):
            state.publish_frame(stream_message(StreamMessageType.FRAME, {"match_time_s": i}))
        assert queue.full(), "the queue did not fill, so nothing is being tested"

        state.publish(stream_message(StreamMessageType.RESTART, {}), critical=True)
        kinds = [m["type"] for m in _messages(queue)]

        # A slow tab must lose frames rather than stall the replay, but losing
        # the restart marker would leave it showing a replay that no longer
        # exists, forever.
        assert "restart" in kinds
        assert kinds.count("frame") < 300

    def test_a_barrier_discards_the_snapshot_picture(self) -> None:
        """A rewind invalidates the retained frame as well as the client's buffer."""
        settings = Settings()
        state = _state(settings, speed=0.0)
        state.announce(stream_message(StreamMessageType.MATCH, {"match_id": "x", "loading": False}))
        state.publish_frame(stream_message(StreamMessageType.FRAME, {"match_time_s": 99.0}))

        state.publish_barrier(stream_message(StreamMessageType.RESTART, {}))

        seeded = [m["type"] for m in _messages(state.subscribe())]
        assert seeded == ["match"], (
            "a visitor arriving after a rewind was seeded with the previous lap's position"
        )


class TestSubscriberSnapshot:
    """A visitor joining a shared replay mid-match."""

    def test_a_new_subscriber_is_seeded_with_the_current_picture(self) -> None:
        settings = Settings()
        state = _state(settings, speed=0.0)
        announcement = {"match_id": "Synthetic_Demo", "loading": False}
        state.announce(stream_message(StreamMessageType.MATCH, announcement))
        state.publish_frame(
            stream_message(StreamMessageType.FRAME, {"match_time_s": 12.0, "lap": 0})
        )

        seeded = list(_messages(state.subscribe()))

        # Without this the visitor watches an empty pitch until the next
        # scheduled picture, with no idea which fixture the frames belong to.
        assert [m["type"] for m in seeded] == ["match", "frame"]
        assert seeded[1]["payload"]["match_time_s"] == 12.0

    def test_only_the_newest_picture_is_retained_without_an_announcement(self) -> None:
        """The public demo's own case: it announces nothing, ever.

        A match message is published only when the match *changes*, and the
        hosted demo never changes match. So the snapshot's first slot stays
        empty for the entire life of the process, and an implementation that
        assumed slot zero held an announcement would quietly start treating the
        previous frame as one — seeding every visitor with a stale position
        followed by the current one, which their interpolator would then blend
        between across the whole gap.
        """
        settings = Settings()
        state = _state(settings, speed=0.0)

        for t in (1.0, 2.0, 3.0):
            state.publish_frame(stream_message(StreamMessageType.FRAME, {"match_time_s": t}))

        seeded = list(_messages(state.subscribe()))
        assert len(seeded) == 1, f"a visitor was seeded with {len(seeded)} messages, not one"
        assert seeded[0]["payload"]["match_time_s"] == 3.0

    def test_the_seeded_match_and_frame_always_agree(self) -> None:
        """The race a two-field snapshot would lose.

        A visitor seeded with an announcement from one fixture and a picture
        from the previous one would draw the old positions under the new name,
        and then interpolate out of them into the first real frame.
        """
        settings = Settings()
        state = _state(settings, speed=0.0)
        state.announce(stream_message(StreamMessageType.MATCH, {"match_id": "first"}))
        state.publish_frame(stream_message(StreamMessageType.FRAME, {"fixture": "first"}))

        state.announce(stream_message(StreamMessageType.MATCH, {"match_id": "second"}))

        seeded = list(_messages(state.subscribe()))
        assert [m["type"] for m in seeded] == ["match"], (
            "the previous fixture's frame survived the announcement of a new one"
        )
        assert seeded[0]["payload"]["match_id"] == "second"

    def test_subscribers_do_not_change_what_the_producer_publishes(self) -> None:
        """Fan-out is per-subscriber; the schedule is not."""
        settings = Settings()
        state = _state(settings, speed=0.0)
        first = state.subscribe()
        second = state.subscribe()

        state.publish_frame(stream_message(StreamMessageType.FRAME, {"match_time_s": 1.0}))

        assert [m["type"] for m in _messages(first)] == ["frame"]
        assert [m["type"] for m in _messages(second)] == ["frame"]


class TestFramePayload:
    """What a visual frame has to carry for the client to buffer it correctly."""

    async def test_a_frame_carries_its_source_identity_and_pacing(self) -> None:
        settings = Settings()
        # A genuinely paced player here, because the payload has to report the
        # speed the replay is actually running at. Four seconds of fixture at
        # 4x is one second of test.
        state = _state(settings, speed=4.0, period_s=4.0)
        queue = state.subscribe()

        await asyncio.wait_for(run_replay(state), timeout=60.0)
        frames = [m["payload"] for m in _messages(queue) if m["type"] == "frame"]

        assert frames, "no visual frame was published at all"
        for payload in frames:
            # Without these the browser cannot order samples, cannot tell one
            # lap from the next, and cannot convert a wall-clock playout delay
            # into the match seconds its render clock advances in.
            assert set(payload) >= {"match_time_s", "frame", "lap", "fixture", "speed", "period"}
            assert payload["fixture"] == SYNTHETIC_MATCH_ID
            assert payload["speed"] == 4.0
            assert payload["lap"] == 0

    async def test_source_time_and_frame_ids_are_monotonic_within_a_lap(self) -> None:
        settings = Settings()
        state = _state(settings, speed=0.0, period_s=6.0)
        queue = state.subscribe()

        await asyncio.wait_for(run_replay(state, clock=_FrameClock(8.0)), timeout=60.0)
        frames = [m["payload"] for m in _messages(queue) if m["type"] == "frame"]

        assert len(frames) > 5
        times = [f["match_time_s"] for f in frames]
        ids = [f["frame"] for f in frames]
        assert times == sorted(times), "source time went backwards on the wire"
        assert ids == sorted(ids), "source frame ids went backwards on the wire"
        assert len(set(ids)) == len(ids), "the same source frame was published twice"


class TestFixtureRotation:
    """The public rotation, and the order its changeover must happen in."""

    @staticmethod
    def _public(period_s: float = 6.0) -> AppState:
        """Public state with a three-fixture rotation, short enough to cycle fast."""
        settings = Settings()
        settings.service.public_demo = True
        state = _state(settings, speed=0.0, period_s=period_s)

        rotation: list[FixtureRotation] = []
        for source in demo_fixture_cycle():
            # The real fixtures, shortened: what is under test is the changeover,
            # and five minutes of tracking per fixture would dominate the suite.
            short = replace(source, period_duration_s=period_s)
            tracking, events, orientation = short.load()
            rotation.append(
                FixtureRotation(
                    match_id=source.match_id,
                    name=source.profile.name,
                    narrative=source.profile.narrative,
                    tracking=tracking,
                    events=events,
                    orientation=orientation,
                )
            )
        state.fixtures = tuple(rotation)

        # Start on the first fixture, as bootstrap does.
        first = state.fixtures[0]
        state.player = ReplayPlayer(
            match_id=first.match_id,
            tracking=first.tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=0.0,
        )
        state.engine = build_engine(
            settings,
            first.tracking,
            first.events,
            first.orientation,
            HeuristicPredictor(settings.model.threshold),
            state.metrics,
        )
        return state

    def test_the_rotation_is_three_distinct_archetypes(self) -> None:
        state = self._public()
        assert len({f.match_id for f in state.fixtures}) == 3
        assert len({f.name for f in state.fixtures}) == 3

    def test_rotation_advances_in_order_and_wraps(self) -> None:
        state = self._public()
        seen = [state.fixtures[state.fixture_index].match_id]
        for _ in range(3):
            seen.append(rotate_fixture(state) or "")

        assert seen[:3] == [f.match_id for f in state.fixtures]
        assert seen[3] == state.fixtures[0].match_id, "the rotation did not wrap"

    def test_the_engine_is_rebuilt_before_the_new_fixture_is_scored(self) -> None:
        """The hazard a changeover shares with a rewind.

        A rolling window still holding five seconds of the previous match would
        be scored as though play were continuous, and the engine's monotonic
        frame check would sit at the end of one fixture while the next starts
        from frame one — every frame of it rejected as out of order.
        """
        state = self._public()
        before = state.engine
        rotate_fixture(state)

        assert state.engine is not before, "the engine survived the changeover"
        assert state.player is not None
        assert state.player.status().match_id == state.fixtures[1].match_id

    def test_the_barrier_precedes_the_announcement_and_the_first_frame(self) -> None:
        state = self._public()
        queue = state.subscribe()
        list(_messages(queue))  # drain the snapshot

        rotate_fixture(state)
        kinds = [m["type"] for m in _messages(queue)]

        # A client clears its interpolation buffer on the barrier, then learns
        # what it is about to draw. Reversing the two would have it blending the
        # last frame of one fixture into the first of the next.
        assert kinds[:2] == ["restart", "match"], kinds

    def test_a_visitor_arriving_mid_changeover_is_not_shown_the_old_fixture(self) -> None:
        state = self._public()
        state.publish_frame(
            stream_message(StreamMessageType.FRAME, {"fixture": "old", "match_time_s": 5.0})
        )
        rotate_fixture(state)

        seeded = list(_messages(state.subscribe()))
        assert [m["type"] for m in seeded] == ["match"]
        assert seeded[0]["payload"]["match_id"] == state.fixtures[1].match_id

    def test_a_single_fixture_does_not_rotate(self) -> None:
        """Outside the public rotation a match changes only when asked."""
        state = _state(Settings(), speed=0.0)
        assert rotate_fixture(state) is None
