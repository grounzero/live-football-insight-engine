"""Canonical domain types shared by every stage of the pipeline.

Both Metrica formats (the CSV of sample games 1-2 and the EPTS/JSON of sample
game 3) are parsed into these types, so nothing downstream needs to know which
source a match came from.

Tracking is held columnar as numpy arrays because feature extraction is
vectorised over whole matches; :class:`Frame` is the per-frame view used by the
live path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Self

import numpy as np

from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import Iterator


class Team(StrEnum):
    """Which side of the match a row of data belongs to."""

    HOME = "home"
    AWAY = "away"

    @property
    def opponent(self) -> Team:
        """The other team."""
        return Team.AWAY if self is Team.HOME else Team.HOME


class AttackDirection(StrEnum):
    """The direction a team attacks in canonical coordinates.

    Canonical coordinates are fixed per match (see :mod:`football_insights.pitch`);
    a team's attacking direction changes between halves, which is exactly what
    this records.
    """

    POSITIVE_X = "+x"
    NEGATIVE_X = "-x"

    @property
    def sign(self) -> float:
        """``+1.0`` when attacking ``+x``, ``-1.0`` otherwise."""
        return 1.0 if self is AttackDirection.POSITIVE_X else -1.0

    @property
    def flipped(self) -> AttackDirection:
        """The opposite direction."""
        return (
            AttackDirection.NEGATIVE_X
            if self is AttackDirection.POSITIVE_X
            else AttackDirection.POSITIVE_X
        )

    @classmethod
    def from_sign(cls, value: float) -> AttackDirection:
        """Build from any positive or negative number."""
        return cls.POSITIVE_X if value >= 0 else cls.NEGATIVE_X


class EventType(StrEnum):
    """Event taxonomy normalised across both Metrica formats.

    Sample game 3 emits ``CARRY`` events that games 1 and 2 do not. That
    asymmetry is preserved here rather than hidden, and is documented in the
    data card; nothing in the label or feature path may depend on ``CARRY``
    being present.
    """

    PASS = "pass"
    CARRY = "carry"
    SHOT = "shot"
    RECOVERY = "recovery"
    BALL_LOST = "ball_lost"
    CHALLENGE = "challenge"
    SET_PIECE = "set_piece"
    BALL_OUT = "ball_out"
    FAULT_RECEIVED = "fault_received"
    CARD = "card"
    OTHER = "other"


#: Event types that indicate the ball is not in open play. Windows overlapping
#: these are excluded from training and suppressed at serving time.
DEAD_BALL_TYPES: Final[frozenset[EventType]] = frozenset(
    {EventType.SET_PIECE, EventType.BALL_OUT, EventType.CARD, EventType.FAULT_RECEIVED}
)

#: Event types that represent a team actively moving the ball. Used to derive
#: the possession approximation.
ON_BALL_TYPES: Final[frozenset[EventType]] = frozenset(
    {
        EventType.PASS,
        EventType.CARRY,
        EventType.SHOT,
        EventType.RECOVERY,
        EventType.BALL_LOST,
    }
)


@dataclass(frozen=True, slots=True)
class PlayerRef:
    """Identity of a tracked player.

    ``is_goalkeeper`` is populated from metadata when the source provides it
    (sample game 3) and inferred otherwise; :mod:`football_insights.data.orientation`
    treats the two cases as different tiers of evidence.
    """

    player_id: str
    team: Team
    shirt_number: int | None = None
    position_type: str | None = None
    is_goalkeeper: bool = False
    goalkeeper_source: str = "unknown"


@dataclass(frozen=True, slots=True)
class Event:
    """A single annotated match event in canonical coordinates.

    Timing carries both the source frame index and seconds; the frame index is
    authoritative because Metrica synchronises tracking and events on it.

    Attributes:
        end_frame: Frame at which the event resolves. For an event still in
            flight at prediction time this lies in the future, which is why
            :class:`~football_insights.features.causal.CausalEventView` refuses
            to expose ``end_*`` fields until it has passed.
    """

    team: Team
    type: EventType
    subtype: str | None
    period: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    from_player: str | None = None
    to_player: str | None = None
    start_xy: tuple[float, float] | None = None
    end_xy: tuple[float, float] | None = None
    raw_type: str = ""

    @property
    def is_dead_ball(self) -> bool:
        """Whether this event marks a stoppage rather than open play."""
        return self.type in DEAD_BALL_TYPES

    @property
    def is_on_ball(self) -> bool:
        """Whether this event represents a team actively moving the ball."""
        return self.type in ON_BALL_TYPES


@dataclass(frozen=True, slots=True)
class Frame:
    """A single tracking frame; the unit consumed by the live path.

    Player arrays keep a stable column order for the whole match, so column
    ``j`` is the same player in every frame. Absent players are ``NaN``.
    """

    period: int
    frame: int
    time_s: float
    home_xy: np.ndarray
    away_xy: np.ndarray
    ball_xy: np.ndarray

    def team_xy(self, team: Team) -> np.ndarray:
        """Positions for one team, shape ``(n_players, 2)``."""
        return self.home_xy if team is Team.HOME else self.away_xy

    @property
    def has_ball(self) -> bool:
        """Whether the ball position is present in this frame."""
        return bool(np.all(np.isfinite(self.ball_xy)))


@dataclass(frozen=True, slots=True)
class MatchTracking:
    """Columnar tracking for one match, in canonical coordinates.

    All arrays share a leading axis of length ``n_frames`` and are ordered by
    ``(period, frame)``. Positions are metres with the origin at the centre
    spot; no per-team reorientation has been applied at this stage.
    """

    period: np.ndarray
    frame: np.ndarray
    time_s: np.ndarray
    home_xy: np.ndarray
    away_xy: np.ndarray
    ball_xy: np.ndarray
    home_players: tuple[PlayerRef, ...]
    away_players: tuple[PlayerRef, ...]
    frame_rate: float

    def __post_init__(self) -> None:
        """Validate array shapes agree; a mismatch here corrupts everything downstream."""
        n = self.period.shape[0]
        expected = {
            "frame": (n,),
            "time_s": (n,),
            "home_xy": (n, len(self.home_players), 2),
            "away_xy": (n, len(self.away_players), 2),
            "ball_xy": (n, 2),
        }
        for name, shape in expected.items():
            got = getattr(self, name).shape
            if got != shape:
                msg = f"MatchTracking.{name} has shape {got}, expected {shape}"
                raise ValueError(msg)

    @property
    def n_frames(self) -> int:
        """Number of tracking frames in the match."""
        return int(self.period.shape[0])

    @property
    def duration_s(self) -> float:
        """Wall-clock span of the tracking data in seconds."""
        if self.n_frames == 0:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])

    def players(self, team: Team) -> tuple[PlayerRef, ...]:
        """Player references for one team, in column order."""
        return self.home_players if team is Team.HOME else self.away_players

    def team_xy(self, team: Team) -> np.ndarray:
        """Positions for one team, shape ``(n_frames, n_players, 2)``."""
        return self.home_xy if team is Team.HOME else self.away_xy

    def frame_at(self, index: int) -> Frame:
        """Materialise a single :class:`Frame` view by row index."""
        return Frame(
            period=int(self.period[index]),
            frame=int(self.frame[index]),
            time_s=float(self.time_s[index]),
            home_xy=self.home_xy[index],
            away_xy=self.away_xy[index],
            ball_xy=self.ball_xy[index],
        )

    def iter_frames(self) -> Iterator[Frame]:
        """Iterate every frame in order."""
        for i in range(self.n_frames):
            yield self.frame_at(i)

    def slice(self, start: int, stop: int) -> Self:
        """Return a row slice sharing the underlying arrays."""
        return type(self)(
            period=self.period[start:stop],
            frame=self.frame[start:stop],
            time_s=self.time_s[start:stop],
            home_xy=self.home_xy[start:stop],
            away_xy=self.away_xy[start:stop],
            ball_xy=self.ball_xy[start:stop],
            home_players=self.home_players,
            away_players=self.away_players,
            frame_rate=self.frame_rate,
        )


@dataclass(frozen=True, slots=True)
class Orientation:
    """Attacking direction per period and team, with its supporting evidence.

    Produced by :mod:`football_insights.data.orientation`. The ``report`` field
    carries the full audit trail written to
    ``artifacts/preprocessing/direction_report.json``.
    """

    directions: dict[tuple[int, Team], AttackDirection]
    report: JsonDict = field(default_factory=dict)

    def direction(self, period: int, team: Team) -> AttackDirection:
        """Attacking direction for one team in one period."""
        try:
            return self.directions[(period, team)]
        except KeyError as exc:
            msg = f"no attacking direction recorded for period {period}, team {team}"
            raise KeyError(msg) from exc

    def periods(self) -> tuple[int, ...]:
        """Every period covered, in ascending order."""
        return tuple(sorted({p for p, _ in self.directions}))


@dataclass(frozen=True, slots=True)
class Match:
    """A fully parsed, validated and oriented match."""

    match_id: str
    tracking: MatchTracking
    events: tuple[Event, ...]
    orientation: Orientation
    source_format: str
    metadata: JsonDict = field(default_factory=dict)

    @property
    def frame_rate(self) -> float:
        """Tracking sample rate in hertz."""
        return self.tracking.frame_rate
