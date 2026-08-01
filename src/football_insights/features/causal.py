"""Causal event access: the structural guarantee against temporal leakage.

Feature builders never receive the raw event list. They receive a
:class:`CausalEventView`, which can only answer questions about a specific
instant and physically cannot hand back information that was not yet observable
at that instant.

Two distinct rules are enforced, because leakage hides in the gap between them:

1. **Visibility.** An event exists only once ``start_frame <= now``. Nothing
   about a future event is reachable, not even that it will happen.
2. **Resolution.** A visible event whose ``end_frame`` is still in the future is
   *in flight*: its type, team and origin are known, but its outcome is not.
   :class:`VisibleEvent` sets ``end_frame``, ``end_time_s``, ``end_xy`` and
   ``to_player`` to ``None`` until it resolves.

Rule 2 is the subtle one. A pass event carries the identity of the player who
eventually receives it; reading that mid-flight would tell the model the pass
succeeds and where it lands. Because the field is absent rather than merely
discouraged, that mistake is a ``None`` at runtime and a type error under mypy,
not something a reviewer has to notice.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from football_insights.domain import DEAD_BALL_TYPES, ON_BALL_TYPES, Event, EventType, Team

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class VisibleEvent:
    """An event as it could legitimately be observed at a given instant.

    Attributes:
        resolved: Whether the event had finished by the observation instant.
            While ``False`` every outcome field is ``None``.
    """

    team: Team
    type: EventType
    subtype: str | None
    period: int
    start_frame: int
    start_time_s: float
    start_xy: tuple[float, float] | None
    from_player: str | None
    resolved: bool
    end_frame: int | None = None
    end_time_s: float | None = None
    end_xy: tuple[float, float] | None = None
    to_player: str | None = None

    @property
    def is_on_ball(self) -> bool:
        """Whether this event represents a team actively moving the ball."""
        return self.type in ON_BALL_TYPES

    @property
    def is_dead_ball(self) -> bool:
        """Whether this event marks a stoppage."""
        return self.type in DEAD_BALL_TYPES


@dataclass(frozen=True, slots=True)
class PossessionState:
    """Possession as it can be known at an instant, with no hindsight.

    Possession is an *approximation* derived from the event annotation: the team
    credited with the most recent on-ball event. Its limitations are documented
    in the data card and model card. In particular the true instant a turnover
    occurs is only known once the next event has started, so this state lags
    reality slightly — which is exactly what a live system would experience.
    """

    team: Team | None
    #: Seconds the current team has held the ball, measured to the observation
    #: instant rather than to the end of the sequence.
    duration_s: float
    start_frame: int | None
    #: Number of on-ball events by the current team so far in this run.
    event_count: int
    #: True when the most recent visible event is a stoppage.
    is_dead_ball: bool
    #: True when the most recent visible event has not yet resolved.
    has_event_in_flight: bool


class CausalEventView:
    """Time-indexed, forward-blind access to a match's events.

    Args:
        events: All match events. They are copied and sorted internally, so the
            caller's ordering does not matter.
        frame_rate: Tracking sample rate, used to convert frames to seconds.
    """

    __slots__ = ("_events", "_frame_rate", "_start_frames")

    def __init__(self, events: Sequence[Event], frame_rate: float) -> None:
        """Index the events for forward-blind access."""
        self._events: tuple[Event, ...] = tuple(sorted(events, key=lambda e: e.start_frame))
        self._start_frames: list[int] = [e.start_frame for e in self._events]
        self._frame_rate = float(frame_rate)

    def __len__(self) -> int:
        """Total number of events in the match, across all time."""
        return len(self._events)

    def _visible_count(self, now_frame: int) -> int:
        """Number of events that have started at or before ``now_frame``."""
        return bisect.bisect_right(self._start_frames, now_frame)

    def _redact(self, event: Event, now_frame: int) -> VisibleEvent:
        """Project an event onto what was observable at ``now_frame``."""
        resolved = event.end_frame <= now_frame
        return VisibleEvent(
            team=event.team,
            type=event.type,
            subtype=event.subtype,
            period=event.period,
            start_frame=event.start_frame,
            start_time_s=event.start_time_s,
            start_xy=event.start_xy,
            from_player=event.from_player,
            resolved=resolved,
            end_frame=event.end_frame if resolved else None,
            end_time_s=event.end_time_s if resolved else None,
            end_xy=event.end_xy if resolved else None,
            to_player=event.to_player if resolved else None,
        )

    def visible(self, now_frame: int, lookback_frames: int | None = None) -> list[VisibleEvent]:
        """Events observable at ``now_frame``, oldest first.

        Args:
            now_frame: The observation instant.
            lookback_frames: If given, only events starting within this many
                frames of ``now_frame`` are returned.

        Returns:
            Redacted events; unresolved ones carry no outcome fields.
        """
        hi = self._visible_count(now_frame)
        lo = 0
        if lookback_frames is not None:
            lo = bisect.bisect_left(self._start_frames, now_frame - lookback_frames, 0, hi)
        return [self._redact(e, now_frame) for e in self._events[lo:hi]]

    def latest(self, now_frame: int) -> VisibleEvent | None:
        """The most recently started event at ``now_frame``, or ``None``."""
        hi = self._visible_count(now_frame)
        if hi == 0:
            return None
        return self._redact(self._events[hi - 1], now_frame)

    def possession(self, now_frame: int) -> PossessionState:
        """Derive possession at ``now_frame`` using only visible events.

        Walks back through the visible on-ball events while they belong to the
        same team, so ``duration_s`` measures how long that team has held the
        ball *up to now* — never to the end of the sequence, which lies in the
        future and is therefore unknowable.

        Args:
            now_frame: The observation instant.

        Returns:
            The possession state, with ``team`` ``None`` before any event.
        """
        hi = self._visible_count(now_frame)
        if hi == 0:
            return PossessionState(None, 0.0, None, 0, False, False)

        latest = self._events[hi - 1]
        in_flight = latest.end_frame > now_frame
        is_dead = latest.type in DEAD_BALL_TYPES

        # Find the most recent on-ball event to attribute possession.
        idx = hi - 1
        while idx >= 0 and self._events[idx].type not in ON_BALL_TYPES:
            idx -= 1
        if idx < 0:
            return PossessionState(None, 0.0, None, 0, is_dead, in_flight)

        team = self._events[idx].team
        # Walk back over the unbroken run of on-ball events by this team.
        # Non-on-ball events (a card, a stoppage) are stepped over without
        # ending the run, but never become its start.
        first = idx
        count = 0
        cursor = idx
        while cursor >= 0:
            ev = self._events[cursor]
            if ev.type not in ON_BALL_TYPES:
                cursor -= 1
                continue
            if ev.team is not team:
                break
            count += 1
            first = cursor
            cursor -= 1

        start_frame = self._events[first].start_frame
        duration_s = max(0.0, (now_frame - start_frame) / self._frame_rate)
        return PossessionState(
            team=team,
            duration_s=duration_s,
            start_frame=start_frame,
            event_count=count,
            is_dead_ball=is_dead,
            has_event_in_flight=in_flight,
        )

    def recent_type_counts(
        self, now_frame: int, lookback_frames: int, team: Team | None = None
    ) -> dict[EventType, int]:
        """Count visible event types started within a recent window.

        Args:
            now_frame: The observation instant.
            lookback_frames: Width of the backward window in frames.
            team: Restrict to one team, or count both when ``None``.

        Returns:
            Counts keyed by event type; absent types are omitted.
        """
        counts: dict[EventType, int] = {}
        for ev in self.visible(now_frame, lookback_frames):
            if team is not None and ev.team is not team:
                continue
            counts[ev.type] = counts.get(ev.type, 0) + 1
        return counts

    def dead_ball_frames(self, n_frames: int) -> np.ndarray:
        """Boolean mask of frames the ball is out of open play.

        A stoppage runs from the start of a dead-ball event until the next
        on-ball event begins. Used to exclude restarts from training samples and
        to suppress insights during breaks in play.

        Args:
            n_frames: Length of the match in tracking frames.

        Returns:
            Boolean array of shape ``(n_frames,)``.
        """
        mask = np.zeros(n_frames, dtype=bool)
        for i, ev in enumerate(self._events):
            if ev.type not in DEAD_BALL_TYPES:
                continue
            start = max(0, ev.start_frame - 1)
            stop = n_frames
            for nxt in self._events[i + 1 :]:
                if nxt.type in ON_BALL_TYPES:
                    stop = max(start, nxt.start_frame - 1)
                    break
            mask[start:stop] = True
        return mask
