"""Paced replay of recorded frames.

Emits historical frames as though they were arriving live, at a configurable
multiple of real time, with a fault profile applied. The fault profile decides
*which* frames are emitted and in what order; the pacing decides only *when*.
Keeping those concerns apart is what makes a 50x accelerated test produce
exactly the stream a 1x run would.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from football_insights.replay.faults import FaultInjector
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from football_insights.config import FaultProfileSettings
    from football_insights.domain import Frame, MatchTracking
    from football_insights.replay.faults import EmittedFrame, FaultSummary


@dataclass(frozen=True, slots=True)
class ReplayStatus:
    """Current state of a replay, surfaced on ``/replay/status``."""

    match_id: str
    profile: str
    seed: int
    speed: float
    running: bool
    paused: bool
    frames_emitted: int
    total_frames: int
    match_time_s: float
    summary: JsonDict

    def to_dict(self) -> JsonDict:
        """Serialisable form."""
        return {
            "match_id": self.match_id,
            "fault_profile": self.profile,
            "seed": self.seed,
            "speed": self.speed,
            "running": self.running,
            "paused": self.paused,
            "frames_emitted": self.frames_emitted,
            "total_frames": self.total_frames,
            "match_time_s": round(self.match_time_s, 2),
            "fault_summary": self.summary,
        }


class ReplayPlayer:
    """Replays a match's frames at a configurable rate.

    Args:
        match_id: Identifier reported in status and logs.
        tracking: The match to replay.
        profile: Fault profile to apply.
        seed: Seed for fault injection; recorded in status and logs.
        speed: Multiple of real time. ``0`` emits as fast as possible.
    """

    def __init__(
        self,
        *,
        match_id: str,
        tracking: MatchTracking,
        profile: FaultProfileSettings,
        seed: int,
        speed: float = 1.0,
    ) -> None:
        """Apply the fault profile up front so the stream is fixed before playback."""
        self._match_id = match_id
        self._tracking = tracking
        self._speed = speed
        self._injector = FaultInjector(profile, seed)
        frames: Sequence[Frame] = list(tracking.iter_frames())
        self._emitted, self._summary = self._injector.apply(frames)
        self._position = 0
        self._running = False
        self._paused = False
        self._match_time_s = 0.0
        #: Pacing anchor: the match time and wall clock that the emission
        #: schedule is measured from. Held on the instance rather than in
        #: ``stream``'s frame so that a control request arriving on another task
        #: can invalidate it — see :meth:`_drop_anchor`.
        self._origin: float | None = None
        self._started = 0.0
        self._laps = 0

    @property
    def total_frames(self) -> int:
        """Number of frames the replay will emit."""
        return len(self._emitted)

    @property
    def laps(self) -> int:
        """How many times a looping replay has wrapped back to the first frame.

        A counter rather than something the consumer infers from frame numbers,
        because a fault profile is allowed to reorder frames: a comparison
        against the previous frame index would read a reordering as a wrap and
        tear down rolling state in the middle of a half.
        """
        return self._laps

    @property
    def summary(self) -> FaultSummary:
        """What the fault profile did to this stream."""
        return self._summary

    @property
    def emitted(self) -> Sequence[EmittedFrame]:
        """The full emitted stream, for offline use and tests."""
        return self._emitted

    def status(self) -> ReplayStatus:
        """Snapshot of replay state."""
        return ReplayStatus(
            match_id=self._match_id,
            profile=self._injector.profile_name,
            seed=self._injector.seed,
            speed=self._speed,
            running=self._running,
            paused=self._paused,
            frames_emitted=self._position,
            total_frames=self.total_frames,
            match_time_s=self._match_time_s,
            summary=self._summary.to_dict(),
        )

    def set_paused(self, paused: bool) -> None:
        """Pause or resume emission without losing position."""
        self._paused = paused

    def _drop_anchor(self) -> None:
        """Forget the pacing anchor so the next frame re-anchors from now."""
        self._origin = None

    def set_speed(self, speed: float) -> None:
        """Change the replay rate. Pacing resets so the change takes effect at once.

        Dropping the anchor is the whole point: it describes a schedule computed
        at the old rate, and dividing the match time replayed so far by a *lower*
        speed would ask the loop to sleep off the difference — 21 s after only
        3 s of playback going 8x to 1x, and proportionally worse the longer the
        replay has run, which reads as the service having hung.
        """
        self._speed = max(0.0, speed)
        self._drop_anchor()

    @property
    def paused(self) -> bool:
        """Whether emission is currently paused."""
        return self._paused

    def stop(self) -> None:
        """Ask a running replay to stop after the current frame."""
        self._running = False

    def reset(self) -> None:
        """Rewind to the beginning and re-anchor the pacing schedule.

        Dropping the anchor is part of rewinding, not a separate courtesy. The
        anchor describes a schedule measured from a match time the replay is
        about to be far behind again, so every frame of the rewound stream looks
        overdue and the loop emits them as fast as it can until it catches back
        up — the same failure :meth:`set_speed` documents, arriving from the
        other direction.
        """
        self._position = 0
        self._match_time_s = 0.0
        self._drop_anchor()

    async def stream(self, loop: bool = False) -> AsyncIterator[EmittedFrame]:
        """Yield frames at the configured pace.

        Args:
            loop: Restart from the beginning when the match ends.

        Yields:
            Emitted frames, including duplicates and out-of-order arrivals.
        """
        self._running = True
        self._drop_anchor()

        while self._running:
            if self._position >= len(self._emitted):
                if not loop:
                    break
                self.reset()
                # Consumers rebuild their own rolling state from this; see
                # `run_replay`, which resets the engine before the first frame
                # of the new lap so it is not rejected as out of order.
                self._laps += 1

            if self._paused:
                await asyncio.sleep(0.05)
                # Re-anchor so the pause is not "caught up" on resume.
                self._drop_anchor()
                continue

            item = self._emitted[self._position]
            self._position += 1
            self._match_time_s = item.frame.time_s

            paced = False
            if self._speed > 0:
                if self._origin is None:
                    self._origin = item.frame.time_s
                    self._started = time.perf_counter()
                target = (item.frame.time_s - self._origin) / self._speed + item.offset_s
                delay = target - (time.perf_counter() - self._started)
                if delay > 0:
                    await asyncio.sleep(delay)
                    paced = True

            # Yield control periodically whenever the pacing sleep did not.
            #
            # That sleep is the only thing returning control to the event loop on
            # the paced path, and it disappears the moment the replay falls
            # behind schedule: `delay` goes negative, nothing is awaited, and
            # resuming an async generator does not reach the scheduler. The loop
            # then runs the entire match in one uninterrupted burst — no frame
            # reaches a subscriber, `/health` and `/ready` do not answer, and a
            # platform watching those decides the container is dead.
            #
            # Falling behind is not an edge case: it is what happens whenever
            # per-frame work exceeds the budget the speed implies, which at 8x
            # is 5 ms a frame and under coverage instrumentation is routinely
            # less. The unpaced branch below has always guarded against this;
            # the paced one needs the same guard for the same reason.
            if not paced and self._position % 64 == 0:
                await asyncio.sleep(0)

            yield item

        self._running = False
