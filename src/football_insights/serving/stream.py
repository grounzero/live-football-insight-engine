"""The replay loop and the messages it publishes.

Everything a subscriber to ``GET /insights/stream`` eventually sees is built
here: frames, insights, the editorial rollup, restart and end markers. Kept
apart from the routing layer because it is the only part that runs *between*
requests, on a task of its own, and because the pacing constants below are one
subject that a reader should not have to find among route handlers.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np

from football_insights.serving.logging import new_correlation_id
from football_insights.serving.messages import StreamMessageType, stream_message
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import Callable

    from football_insights.domain import Frame
    from football_insights.insight.types import EditorialOutcome, Insight
    from football_insights.serving.engine import EngineResult, InsightEngine
    from football_insights.serving.state import AppState

LOGGER = logging.getLogger("football_insights.serving")

#: Tracking rate the replay path assumes when turning frame counts into match
#: time. The suppression rollup derives from it, so a retune moves both together.
SOURCE_FRAME_HZ: Final = 25.0

#: Visual frames per second of *wall clock*, for every subscriber and at every
#: replay speed.
#:
#: The distinction matters more than it looks. This used to be expressed as a
#: fraction of the source rate and applied as ``counter % publish_every``, which
#: is a ratio of *source frames* rather than a rate: at the hosted demo's pace it
#: multiplied out to a hundred messages a second, which no display can show and
#: no browser should be asked to parse. A schedule measured against the wall
#: clock is the same number at 1x and at 8x.
#:
#: 20 Hz rather than 60: the browser interpolates between these samples on its
#: own refresh, so this only has to be dense enough to describe the motion, and
#: every message above that is bandwidth and main-thread work nobody can see.
VISUAL_PUBLISH_HZ: Final = 20.0

#: Editorial review runs on every processed frame, so one message per decision
#: would be 25 a second whose individual values tell a reader nothing. The frame
#: payload carries the current reason, which answers "why is nothing on screen";
#: the mix of reasons over a short window is published separately, aggregated.
#:
#: It is aggregated *here* rather than in the browser because frames are
#: published at half the source rate: totals derived from the frame stream would
#: be a sample presented as a total. One rollup per second of match time is a
#: twelfth of the frame traffic and is exact.
SUPPRESSION_ROLLUP_FRAMES: Final = int(SOURCE_FRAME_HZ)


@dataclass
class SuppressionRollup:
    """Exact per-reason editorial counts over a fixed number of processed frames.

    An object rather than three locals in the replay loop because the counts,
    the emitted tally and the interval length are one fact: a reader checking
    that a published rollup is exact should not first have to prove that three
    separate variables are always reset together.
    """

    frames: int = 0
    emitted: int = 0
    counts: dict[str, int] = field(default_factory=dict[str, int])

    def record(self, outcome: EditorialOutcome | None) -> str | None:
        """Count one editorial decision, returning its suppression reason if any.

        Frames with no outcome at all — rejected transport, or possession not yet
        resolved — still count toward ``frames``. That keeps the denominator the
        number of frames actually reviewed, so emitted plus suppressed can be
        compared against it honestly instead of summing to a different total.
        """
        self.frames += 1
        if outcome is None:
            return None
        if outcome.insight is not None:
            self.emitted += 1
            return None
        if outcome.suppressed is None:
            return None
        reason = outcome.suppressed.reason.value
        self.counts[reason] = self.counts.get(reason, 0) + 1
        return reason

    def due(self) -> bool:
        """Whether a full rollup interval has been reviewed."""
        return self.frames >= SUPPRESSION_ROLLUP_FRAMES

    def drain(self) -> JsonDict:
        """Return this interval's totals and begin a new one.

        Reasons that did not occur are omitted rather than sent as zeros; the
        client accumulates, so an absent key and a zero mean the same thing and
        the smaller message is the honest one.
        """
        payload: JsonDict = {
            "frames": self.frames,
            "emitted": self.emitted,
            "counts": dict(self.counts),
        }
        self.frames = 0
        self.emitted = 0
        self.counts.clear()
        return payload


class VisualRateLimiter:
    """A monotonic wall-clock schedule for visual frames.

    Answers one question — may a picture go out now? — against a deadline that
    advances on its own clock rather than on the frames arriving. That is the
    whole point: the caller is iterating a replay whose rate is a *multiple* of
    real time, so anything derived from the frames themselves inherits that
    multiple.

    Missed deadlines are skipped, never repaid. A producer that stalls for
    300 ms has six overdue slots when it wakes; publishing six frames back to
    back would put five stale pictures on the wire ahead of the only one anybody
    can still use, and the browser would show the burst as a snap. Advancing the
    deadline past all six and sending the newest frame once is both cheaper and
    the only version that looks right.

    Args:
        hz: Visual frames per second of wall clock.
        clock: Monotonic time source, injectable so tests can drive the schedule
            without sleeping.
    """

    __slots__ = ("_clock", "_deadline", "_period", "_skipped")

    def __init__(
        self, hz: float = VISUAL_PUBLISH_HZ, clock: Callable[[], float] | None = None
    ) -> None:
        """Start unarmed, so the first frame offered is always published."""
        self._period = 1.0 / hz
        self._clock = clock or time.monotonic
        self._deadline: float | None = None
        self._skipped = 0

    @property
    def skipped_deadlines(self) -> int:
        """Deadlines passed over because the producer was late.

        Reported rather than merely counted: a rising number is the signal that
        the replay is not keeping its own schedule, which is a different fault
        from the network being slow and needs a different fix.
        """
        return self._skipped

    def reset(self) -> None:
        """Forget the schedule, so the next frame offered publishes immediately.

        Used at every barrier. A restart, a lap wrap or a period change must put
        a picture on the wire at once — a client that has just cleared its
        interpolation buffer has nothing to draw until it does — and making it
        wait out a deadline set by the previous fixture would be an arbitrary
        blank frame at the one moment the viewer is most likely to notice.
        """
        self._deadline = None

    def due(self, *, force: bool = False) -> bool:
        """Whether a visual frame may be published now.

        Args:
            force: Publish regardless, and re-anchor the schedule from now. For
                barriers the client must see immediately — a period change, say —
                where simply returning True would leave the old deadline standing
                and put a second picture out microseconds later.

        Returns:
            True at most once per period. The caller passes the newest frame it
            has; frames offered between deadlines are dropped for display only
            and still reach inference.
        """
        now = self._clock()
        if force or self._deadline is None:
            self._deadline = now + self._period
            return True
        if now < self._deadline:
            return False
        # How many whole periods have elapsed since the deadline we just missed.
        # `+ 1` because passing the deadline at all consumes the slot it names.
        missed = math.floor((now - self._deadline) / self._period) + 1
        self._skipped += missed - 1
        self._deadline += missed * self._period
        return True


def round_positions(xy: np.ndarray) -> list[list[float] | None]:
    """Round coordinates for transport, replacing absent players with ``None``."""
    out: list[list[float] | None] = []
    for row in np.atleast_2d(xy):
        if not np.all(np.isfinite(row)):
            out.append(None)
        else:
            out.append([round(float(row[0]), 2), round(float(row[1]), 2)])
    return out


@dataclass(frozen=True, slots=True)
class VisualContext:
    """Which replay a visual frame belongs to, and how fast it is running.

    Carried on every frame rather than announced once, because the browser
    interpolates between frames and must never interpolate across a boundary.
    Inferring one from a separate control message would make that correctness
    depend on message ordering across two cadences; putting it in the frame
    makes each frame self-describing, and a client that missed the barrier still
    cannot blend two fixtures together.
    """

    match_id: str
    lap: int
    speed: float


def frame_payload(
    frame: Frame,
    result: EngineResult,
    suppression: str | None,
    context: VisualContext,
) -> JsonDict:
    """Build the browser-facing message describing one frame.

    Every field added here is additive: a client built against the older schema
    ignores what it does not recognise and keeps working.
    """
    probability = (
        result.prediction.probability
        if result.prediction and result.prediction.window_valid
        else None
    )
    return {
        "period": frame.period,
        "match_time_s": round(frame.time_s, 2),
        # Source identity and pacing, for the client's playout buffer. `frame`
        # orders samples that share a timestamp, `lap` and `fixture` bound
        # interpolation, and `speed` converts a wall-clock playout delay into the
        # match seconds the render clock actually advances in.
        "frame": int(frame.frame),
        "lap": context.lap,
        "fixture": context.match_id,
        "speed": context.speed,
        "home": round_positions(frame.home_xy),
        "away": round_positions(frame.away_xy),
        "ball": round_positions(frame.ball_xy[None, :])[0],
        "probability": None if probability is None else round(probability, 4),
        "window_valid": bool(result.prediction.window_valid if result.prediction else False),
        "attacking_team": result.attacking_team,
        # Resolved to a boolean in canonical coordinates rather than shipping the
        # `AttackDirection` string, so the enum's meaning stays server-side and
        # the browser never reimplements the sign convention.
        "attacking_right": (None if result.attacking_sign is None else result.attacking_sign > 0),
        "suppression": suppression,
    }


def publish_insight(state: AppState, insight: Insight) -> None:
    """Record an emitted insight and fan it out to subscribers."""
    state.recent_insights.append(insight)
    state.publish(stream_message(StreamMessageType.INSIGHT, insight.to_dict()))
    LOGGER.info(
        "insight emitted",
        extra={
            "kind": insight.kind.value,
            "probability": round(insight.probability, 3),
            "match_time_s": round(insight.match_time_s, 1),
            "is_ml": insight.is_ml,
        },
    )


def apply_restart(state: AppState, engine: InsightEngine) -> None:
    """Rebuild every piece of per-replay state after a rewind.

    Published as critical: a client that misses this keeps showing insights from
    a replay that no longer exists.
    """
    engine.reset()
    state.recent_insights.clear()
    state.publish_barrier(stream_message(StreamMessageType.RESTART, {}))
    LOGGER.info("replay restarted")


def announce_match(state: AppState, match_id: str, *, loading: bool) -> None:
    """Tell every client which match is being played, and whether it is ready.

    Critical, and sent before the load as well as after it: a client that misses
    the first message sits watching a frozen pitch with no explanation, and one
    that misses the second never stops waiting.
    """
    state.announce(
        stream_message(StreamMessageType.MATCH, {"match_id": match_id, "loading": loading})
    )


def _publish_results(
    state: AppState,
    frame: Frame,
    result: EngineResult,
    rollup: SuppressionRollup,
    context: VisualContext,
    *,
    publish_frame: bool,
) -> None:
    """Fan out everything one processed frame produced.

    Three separate cadences, which is why they are here rather than inline in
    the loop: an insight goes out the moment it is emitted, the editorial rollup
    once per second of match time, and the picture on a wall-clock schedule that
    is deliberately unrelated to either.

    Only the picture is rate-limited. An insight is the product and a rollup is
    an exact total over frames that were all reviewed; dropping either to save
    bandwidth would turn a measurement into a sample presented as a total.
    """
    reason = rollup.record(result.outcome)

    if result.outcome is not None and result.outcome.insight is not None:
        publish_insight(state, result.outcome.insight)

    if rollup.due():
        state.publish(stream_message(StreamMessageType.SUPPRESSION, rollup.drain()))

    if publish_frame:
        state.publish_frame(
            stream_message(StreamMessageType.FRAME, frame_payload(frame, result, reason, context))
        )


async def run_replay(state: AppState, clock: Callable[[], float] | None = None) -> None:
    """Drive the replay through the engine and publish results.

    Args:
        state: Shared application state; supplies the player, the engine and the
            subscriber set.
        clock: Monotonic time source for the visual publish schedule. Injectable
            so a test can assert the cadence without running in real time.
    """
    player = state.player
    engine = state.engine
    if player is None or engine is None:
        return
    status = player.status()
    cid = new_correlation_id()
    LOGGER.info(
        "replay started",
        extra={
            "match_id": status.match_id,
            "fault_profile": status.profile,
            "seed": status.seed,
            "speed": status.speed,
            "visual_publish_hz": VISUAL_PUBLISH_HZ,
            "correlation_id": cid,
        },
    )
    visual = VisualRateLimiter(clock=clock)
    rollup = SuppressionRollup()
    counter = 0
    laps = player.laps
    match_id = status.match_id
    period: int | None = None
    try:
        async for emitted in player.stream(loop=state.settings.replay.loop):
            if state.take_restart():
                # This frame was taken from the old position before the request
                # arrived. Processing it would leave the engine's monotonic frame
                # check ahead of everything about to be replayed, and every frame
                # of the new run would be rejected as out of order — while the
                # pitch kept animating, because frames are published whether or
                # not the engine accepted them. A dead demo that looks alive.
                apply_restart(state, engine)
                rollup = SuppressionRollup()
                counter = 0
                laps = player.laps
                # A barrier the client answers by clearing its interpolation
                # buffer, so it has nothing to draw until the next picture. The
                # schedule is dropped rather than honoured so that picture is the
                # very next frame.
                visual.reset()
                continue

            if player.laps != laps:
                # A looping replay has wrapped. Same hazard as a restart, arriving
                # without anyone asking: the engine's last frame id is at the end
                # of the match and every frame of the new lap would be dropped as
                # out of order. Unlike a restart there is no stale frame to
                # discard — this one is the first frame of the new lap — so the
                # state is rebuilt and the frame then falls through to be
                # processed rather than skipped.
                laps = player.laps
                apply_restart(state, engine)
                rollup = SuppressionRollup()
                counter = 0
                visual.reset()

            if state.should_stop_unwatched():
                LOGGER.info(
                    "no subscribers; stopping the replay until someone connects",
                    extra={"frames": counter},
                )
                return

            counter += 1
            frame = emitted.frame

            # A period change is a barrier for the same reason a lap is: the
            # teams have swapped ends, so interpolating from the last frame of
            # one half into the first of the next would walk twenty-two players
            # across the halfway line over 50 ms.
            barrier = period is not None and frame.period != period
            period = frame.period

            _publish_results(
                state,
                frame,
                engine.process(frame),
                rollup,
                VisualContext(match_id=match_id, lap=laps, speed=player.speed),
                publish_frame=visual.due(force=barrier),
            )
    except asyncio.CancelledError:  # pragma: no cover - shutdown
        raise
    finally:
        # Not published when the loop is being cancelled on purpose: clients
        # close their stream for good on an end marker, so announcing one during
        # a match swap would present a change of match as the end of the match.
        if state.end_marker_wanted:
            state.publish(stream_message(StreamMessageType.END, {"frames": counter}), critical=True)
        LOGGER.info("replay finished", extra={"frames": counter})
