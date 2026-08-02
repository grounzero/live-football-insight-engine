"""The replay loop and the messages it publishes.

Everything a subscriber to ``GET /insights/stream`` eventually sees is built
here: frames, insights, the editorial rollup, restart and end markers. Kept
apart from the routing layer because it is the only part that runs *between*
requests, on a task of its own, and because the pacing constants below are one
subject that a reader should not have to find among route handlers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np

from football_insights.serving.logging import new_correlation_id
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from football_insights.domain import Frame
    from football_insights.insight.types import EditorialOutcome, Insight
    from football_insights.serving.engine import EngineResult, InsightEngine
    from football_insights.serving.state import AppState

LOGGER = logging.getLogger("football_insights.serving")

#: Tracking rate the replay path assumes when turning frame counts into match
#: time. Both the publish throttle and the suppression rollup derive from it, so
#: the two cadences cannot drift apart if either is retuned.
SOURCE_FRAME_HZ: Final = 25.0

#: Frames are sent to the browser at this rate regardless of tracking rate;
#: 25 Hz of JSON per client is wasteful and invisible to the eye.
FRAME_PUBLISH_HZ: Final = 12.5

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


def round_positions(xy: np.ndarray) -> list[list[float] | None]:
    """Round coordinates for transport, replacing absent players with ``None``."""
    out: list[list[float] | None] = []
    for row in np.atleast_2d(xy):
        if not np.all(np.isfinite(row)):
            out.append(None)
        else:
            out.append([round(float(row[0]), 2), round(float(row[1]), 2)])
    return out


def frame_payload(frame: Frame, result: EngineResult, suppression: str | None) -> JsonDict:
    """Build the browser-facing message describing one frame."""
    probability = (
        result.prediction.probability
        if result.prediction and result.prediction.window_valid
        else None
    )
    return {
        "period": frame.period,
        "match_time_s": round(frame.time_s, 2),
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
    state.publish(json.dumps({"type": "insight", "payload": insight.to_dict()}))
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
    state.publish(json.dumps({"type": "restart", "payload": {}}), critical=True)
    LOGGER.info("replay restarted")


def announce_match(state: AppState, match_id: str, *, loading: bool) -> None:
    """Tell every client which match is being played, and whether it is ready.

    Critical, and sent before the load as well as after it: a client that misses
    the first message sits watching a frozen pitch with no explanation, and one
    that misses the second never stops waiting.
    """
    state.publish(
        json.dumps({"type": "match", "payload": {"match_id": match_id, "loading": loading}}),
        critical=True,
    )


def _publish_results(
    state: AppState,
    frame: Frame,
    result: EngineResult,
    rollup: SuppressionRollup,
    *,
    publish_frame: bool,
) -> None:
    """Fan out everything one processed frame produced.

    Three separate cadences, which is why they are here rather than inline in
    the loop: an insight goes out the moment it is emitted, the editorial rollup
    once per second of match time, and the frame itself at half the rate it was
    scored at.
    """
    reason = rollup.record(result.outcome)

    if result.outcome is not None and result.outcome.insight is not None:
        publish_insight(state, result.outcome.insight)

    if rollup.due():
        state.publish(json.dumps({"type": "suppression", "payload": rollup.drain()}))

    if publish_frame:
        state.publish(
            json.dumps({"type": "frame", "payload": frame_payload(frame, result, reason)})
        )


async def run_replay(state: AppState) -> None:
    """Drive the replay through the engine and publish results."""
    player = state.player
    engine = state.engine
    if player is None or engine is None:
        return
    cid = new_correlation_id()
    LOGGER.info(
        "replay started",
        extra={
            "match_id": player.status().match_id,
            "fault_profile": player.status().profile,
            "seed": player.status().seed,
            "speed": player.status().speed,
            "correlation_id": cid,
        },
    )
    publish_every = max(1, round(SOURCE_FRAME_HZ / FRAME_PUBLISH_HZ))
    rollup = SuppressionRollup()
    counter = 0
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
                continue

            counter += 1
            _publish_results(
                state,
                emitted.frame,
                engine.process(emitted.frame),
                rollup,
                publish_frame=counter % publish_every == 0,
            )
    except asyncio.CancelledError:  # pragma: no cover - shutdown
        raise
    finally:
        # Not published when the loop is being cancelled on purpose: clients
        # close their stream for good on an end marker, so announcing one during
        # a match swap would present a change of match as the end of the match.
        if state.end_marker_wanted:
            state.publish(
                json.dumps({"type": "end", "payload": {"frames": counter}}), critical=True
            )
        LOGGER.info("replay finished", extra={"frames": counter})
