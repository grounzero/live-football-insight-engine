"""Deterministic synthetic match generator.

The Metrica sample data carries no formal licence and is far too large to
commit, so every automated test runs against synthetic matches produced here.
They are generated from a seed, so fixtures are reproducible without being
stored as binary blobs.

The simulation is deliberately crude football rather than a physics engine. It
needs to produce three things the pipeline genuinely depends on:

* possession sequences that progress upfield and sometimes enter the penalty
  area, so the label builder has positives to find;
* a defensive shape that responds to the ball, so features such as defensive
  line height and compactness carry signal rather than noise;
* an event stream consistent with the tracking, so event/frame alignment and
  the causal possession view can be exercised.

It also writes real Metrica-format CSV, which lets the production parser be
tested against the exact on-disk layout it will meet in the wild.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from football_insights.domain import (
    AttackDirection,
    Event,
    EventType,
    MatchTracking,
    Orientation,
    PlayerRef,
    Team,
)
from football_insights.pitch import DEFAULT_PITCH, Pitch
from football_insights.types import FloatArray

if TYPE_CHECKING:
    from collections.abc import Sequence

#: 4-4-2 template as fractions of half-length and half-width, for a team
#: attacking +x. Index 0 is the goalkeeper.
FORMATION_442: Final[tuple[tuple[float, float], ...]] = (
    (-0.94, 0.00),  # GK
    (-0.55, -0.55),
    (-0.60, -0.18),
    (-0.60, 0.18),
    (-0.55, 0.55),
    (-0.10, -0.62),
    (-0.15, -0.20),
    (-0.15, 0.20),
    (-0.10, 0.62),
    (0.30, -0.22),
    (0.30, 0.22),
)

MAX_PLAYER_SPEED_MS: Final = 8.0
MAX_BALL_SPEED_MS: Final = 22.0


@dataclass(frozen=True, slots=True)
class SyntheticMatch:
    """A generated match plus the ground truth used to assert against it."""

    tracking: MatchTracking
    events: tuple[Event, ...]
    orientation: Orientation
    #: Times at which the ball genuinely crossed into the attacking penalty
    #: area, with the attacking team. Independent of the label builder, so
    #: tests can check the label builder rather than assume it.
    true_entries: tuple[tuple[float, Team], ...]
    frame_rate: float


@dataclass(slots=True)
class _Sequence:
    """One possession sequence in the simulation."""

    team: Team
    start_s: float
    duration_s: float
    reaches_box: bool
    waypoints: list[tuple[float, float]]


def _formation_targets(
    ball_xy: np.ndarray,
    attacking: bool,
    direction_sign: float,
    pitch: Pitch,
) -> np.ndarray:
    """Target positions for one team given the current ball position.

    The whole block slides up and down the pitch with the ball and squeezes
    toward the ball laterally, which is what makes compactness and line-height
    features meaningful.

    Args:
        ball_xy: Ball position in canonical metres.
        attacking: Whether this team is in possession.
        direction_sign: ``+1`` if this team attacks ``+x``, ``-1`` otherwise.
        pitch: Pitch dimensions.

    Returns:
        Array of shape ``(11, 2)`` in canonical metres.
    """
    template = np.asarray(FORMATION_442, dtype=np.float64)
    xy = np.empty_like(template)
    xy[:, 0] = template[:, 0] * pitch.half_length
    xy[:, 1] = template[:, 1] * pitch.half_width

    # Orient the template to this team's attacking direction.
    xy *= direction_sign

    ball_progress = ball_xy[0] * direction_sign  # +ve when ball is upfield

    # Attackers push higher; defenders hold a line relative to the ball.
    shift = 0.55 * ball_progress if attacking else 0.45 * ball_progress + 8.0
    xy[1:, 0] += shift * direction_sign

    # Lateral squeeze toward the ball, stronger for the defending side.
    squeeze = 0.25 if attacking else 0.42
    xy[1:, 1] = xy[1:, 1] * (1.0 - squeeze) + ball_xy[1] * squeeze

    # Goalkeeper stays home, tracking the ball's lateral position a little.
    xy[0, 0] = -0.96 * pitch.half_length * direction_sign
    xy[0, 1] = 0.22 * ball_xy[1]

    np.clip(xy[:, 0], -pitch.half_length + 1.0, pitch.half_length - 1.0, out=xy[:, 0])
    np.clip(xy[:, 1], -pitch.half_width + 1.0, pitch.half_width - 1.0, out=xy[:, 1])
    return xy


def _step_toward(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move ``current`` toward ``target`` under a per-frame distance cap."""
    delta = target - current
    dist = np.hypot(delta[:, 0], delta[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(dist > max_step, max_step / np.maximum(dist, 1e-9), 1.0)
    return np.asarray(current + delta * scale[:, None])


def _build_sequences(
    rng: random.Random,
    period_duration_s: float,
    box_entry_rate: float,
) -> list[_Sequence]:
    """Lay out possession sequences across one period."""
    sequences: list[_Sequence] = []
    t = 0.0
    team = Team.HOME
    while t < period_duration_s - 15.0:
        duration = rng.uniform(6.0, 22.0)
        reaches_box = rng.random() < box_entry_rate
        sequences.append(
            _Sequence(
                team=team,
                start_s=t,
                duration_s=duration,
                reaches_box=reaches_box,
                waypoints=[],
            )
        )
        # A short dead-ball gap between sequences keeps restarts in the data.
        t += duration + rng.uniform(1.0, 4.0)
        # Possession usually turns over, but not always.
        if rng.random() < 0.75:
            team = team.opponent
    return sequences


def _sequence_waypoints(
    seq: _Sequence,
    rng: random.Random,
    direction_sign: float,
    pitch: Pitch,
) -> list[tuple[float, float]]:
    """Ball waypoints from deep possession to either a box entry or a breakdown."""
    n_legs = max(2, int(seq.duration_s / 3.0))
    start_x = rng.uniform(-0.75, -0.25) * pitch.half_length * direction_sign
    start_y = rng.uniform(-0.6, 0.6) * pitch.half_width

    if seq.reaches_box:
        # Finish comfortably inside the penalty area.
        end_x = rng.uniform(0.75, 0.92) * pitch.half_length * direction_sign
        end_y = rng.uniform(-0.5, 0.5) * pitch.penalty_area_y_abs_max
    else:
        # Break down somewhere short of the box.
        end_x = rng.uniform(0.05, 0.60) * pitch.half_length * direction_sign
        end_y = rng.uniform(-0.8, 0.8) * pitch.half_width

    points: list[tuple[float, float]] = []
    for i in range(n_legs + 1):
        f = i / n_legs
        # Ease-in so the ball accelerates upfield late in the move.
        eased = f * f * (3.0 - 2.0 * f)
        x = start_x + (end_x - start_x) * eased
        y = start_y + (end_y - start_y) * eased
        if 0 < i < n_legs:
            x += rng.uniform(-4.0, 4.0)
            y += rng.uniform(-9.0, 9.0)
        y = max(-pitch.half_width + 2.0, min(pitch.half_width - 2.0, y))
        points.append((x, y))
    return points


def _period_directions(n_periods: int) -> dict[tuple[int, Team], AttackDirection]:
    """Attacking direction per period and team.

    Home attacks ``+x`` in period 1 and flips at half time, mirroring real data.
    """
    directions: dict[tuple[int, Team], AttackDirection] = {}
    for period in range(1, n_periods + 1):
        home_dir = AttackDirection.POSITIVE_X if period % 2 == 1 else AttackDirection.NEGATIVE_X
        directions[(period, Team.HOME)] = home_dir
        directions[(period, Team.AWAY)] = home_dir.flipped
    return directions


def _sequence_at(
    sequences: Sequence[_Sequence], seq_index: int, t_rel: float
) -> tuple[int, _Sequence | None]:
    """Advance past finished sequences and return the one covering ``t_rel``.

    Returns:
        The updated index and the active sequence, or ``None`` between
        sequences — the dead-ball gaps the label builder must exclude.
    """
    while (
        seq_index < len(sequences)
        and t_rel > sequences[seq_index].start_s + sequences[seq_index].duration_s
    ):
        seq_index += 1
    if seq_index < len(sequences) and t_rel >= sequences[seq_index].start_s:
        return seq_index, sequences[seq_index]
    return seq_index, None


def _advance_ball(ball: FloatArray, active: _Sequence, t_rel: float, dt: float) -> FloatArray:
    """Move the ball one frame along the active sequence's waypoints.

    Progress through the sequence is linear in time; the waypoints themselves
    carry the ease-in, so the ball accelerates upfield late in a move. Speed is
    capped so the resulting velocity features stay physically plausible.
    """
    f = (t_rel - active.start_s) / active.duration_s
    f = min(max(f, 0.0), 1.0)
    leg = f * (len(active.waypoints) - 1)
    lo = math.floor(leg)
    hi = min(lo + 1, len(active.waypoints) - 1)
    frac = leg - lo
    target = np.array(
        [
            active.waypoints[lo][0] * (1 - frac) + active.waypoints[hi][0] * frac,
            active.waypoints[lo][1] * (1 - frac) + active.waypoints[hi][1] * frac,
        ]
    )
    delta = target - ball
    dist = float(np.hypot(*delta))
    cap = MAX_BALL_SPEED_MS * dt
    moved: FloatArray = ball + delta * (cap / dist if dist > cap else 1.0)
    return moved


@dataclass(frozen=True, slots=True)
class _PeriodClock:
    """Where one period sits in the match timeline.

    Frame numbers and timestamps run continuously across periods, so simulating
    a period needs its offsets as well as its length. Grouping them keeps the
    five values that always travel together from being five parameters.
    """

    period: int
    n_frames: int
    #: Frames emitted by earlier periods; source frame numbers continue from here.
    frame_offset: int
    #: Seconds elapsed in earlier periods.
    start_s: float
    frame_rate: float

    @property
    def dt(self) -> float:
        """Seconds per frame."""
        return 1.0 / self.frame_rate


@dataclass(slots=True)
class _PeriodFrames:
    """Tracking columns for one simulated period."""

    period: np.ndarray
    frame: np.ndarray
    time_s: np.ndarray
    home_xy: np.ndarray
    away_xy: np.ndarray
    ball_xy: np.ndarray
    #: Ball crossings into the attacking penalty area, in absolute match time.
    true_entries: list[tuple[float, Team]]


def _simulate_period(
    sequences: Sequence[_Sequence],
    directions: dict[tuple[int, Team], AttackDirection],
    clock: _PeriodClock,
    pitch: Pitch,
) -> _PeriodFrames:
    """Step the ball and both teams through one period, frame by frame.

    Draws no random numbers: the randomness is entirely in the sequences and
    waypoints handed in, which keeps the seed-to-output mapping easy to reason
    about.
    """
    period, n_frames, dt = clock.period, clock.n_frames, clock.dt
    n_players = len(FORMATION_442)
    home_pos = _formation_targets(np.zeros(2), True, directions[(period, Team.HOME)].sign, pitch)
    away_pos = _formation_targets(np.zeros(2), False, directions[(period, Team.AWAY)].sign, pitch)

    out = _PeriodFrames(
        period=np.full(n_frames, period, dtype=np.int16),
        frame=np.arange(clock.frame_offset + 1, clock.frame_offset + 1 + n_frames, dtype=np.int64),
        time_s=clock.start_s + np.arange(1, n_frames + 1, dtype=np.float64) * dt,
        home_xy=np.full((n_frames, n_players, 2), np.nan),
        away_xy=np.full((n_frames, n_players, 2), np.nan),
        ball_xy=np.full((n_frames, 2), np.nan),
        true_entries=[],
    )

    seq_index = 0
    ball = np.zeros(2)
    in_box_prev = False

    for i in range(n_frames):
        t_rel = (i + 1) * dt
        seq_index, active = _sequence_at(sequences, seq_index, t_rel)

        if active is None:
            # Dead ball: drift the ball gently toward the centre.
            ball = ball * 0.98
            attacking_team = Team.HOME
            in_possession = False
        else:
            attacking_team = active.team
            in_possession = True
            ball = _advance_ball(ball, active, t_rel, dt)

        step = MAX_PLAYER_SPEED_MS * dt
        home_pos = _step_toward(
            home_pos,
            _formation_targets(
                ball,
                attacking_team is Team.HOME and in_possession,
                directions[(period, Team.HOME)].sign,
                pitch,
            ),
            step,
        )
        away_pos = _step_toward(
            away_pos,
            _formation_targets(
                ball,
                attacking_team is Team.AWAY and in_possession,
                directions[(period, Team.AWAY)].sign,
                pitch,
            ),
            step,
        )

        out.home_xy[i] = home_pos
        out.away_xy[i] = away_pos
        out.ball_xy[i] = ball

        # A genuine box entry is the ball crossing into the attacking penalty
        # area while that team is in possession — the ground truth the label
        # builder is checked against, so it is derived independently of it.
        if not in_possession:
            in_box_prev = False
            continue
        oriented = ball * directions[(period, attacking_team)].sign
        inside = bool(pitch.is_inside_penalty_area(oriented))
        if inside and not in_box_prev:
            out.true_entries.append((clock.start_s + t_rel, attacking_team))
        in_box_prev = inside

    return out


def _sequence_events(seq: _Sequence, rng: random.Random, clock: _PeriodClock) -> list[Event]:
    """Build the event stream for one possession sequence.

    One event per waypoint leg plus a terminal event, timed and positioned from
    the same waypoints the tracking followed, so events and frames agree.
    """
    events: list[Event] = []
    period, frame_offset, start_s, frame_rate = (
        clock.period,
        clock.frame_offset,
        clock.start_s,
        clock.frame_rate,
    )
    n_legs = len(seq.waypoints) - 1
    leg_dur = seq.duration_s / max(n_legs, 1)
    for k in range(n_legs):
        s_rel = seq.start_s + k * leg_dur
        e_rel = s_rel + leg_dur * 0.7
        kind = EventType.PASS if k % 3 != 2 else EventType.CARRY
        events.append(
            Event(
                team=seq.team,
                type=kind,
                subtype=None,
                period=period,
                start_frame=frame_offset + int(s_rel * frame_rate) + 1,
                end_frame=frame_offset + int(e_rel * frame_rate) + 1,
                start_time_s=start_s + s_rel,
                end_time_s=start_s + e_rel,
                from_player=f"{seq.team.value}_{(k % 10) + 2}",
                to_player=f"{seq.team.value}_{((k + 3) % 10) + 2}",
                start_xy=seq.waypoints[k],
                end_xy=seq.waypoints[k + 1],
                raw_type=kind.value.upper(),
            )
        )

    end_rel = seq.start_s + seq.duration_s
    terminal = EventType.SHOT if seq.reaches_box and rng.random() < 0.45 else EventType.BALL_LOST
    events.append(
        Event(
            team=seq.team,
            type=terminal,
            subtype="ON TARGET-SAVED" if terminal is EventType.SHOT else "INTERCEPTION",
            period=period,
            start_frame=frame_offset + int(end_rel * frame_rate),
            end_frame=frame_offset + int(end_rel * frame_rate),
            start_time_s=start_s + end_rel,
            end_time_s=start_s + end_rel,
            from_player=f"{seq.team.value}_9",
            to_player=None,
            start_xy=seq.waypoints[-1],
            end_xy=seq.waypoints[-1],
            raw_type=terminal.value.upper(),
        )
    )
    return events


def _player_refs(team: Team, n_players: int) -> tuple[PlayerRef, ...]:
    """Squad references for one team; column 0 is the goalkeeper."""
    return tuple(
        PlayerRef(
            player_id=f"{team.value}_{i + 1}",
            team=team,
            shirt_number=i + 1,
            position_type="Goalkeeper" if i == 0 else None,
            is_goalkeeper=i == 0,
            goalkeeper_source="synthetic",
        )
        for i in range(n_players)
    )


def generate_synthetic_match(
    *,
    seed: int = 7,
    n_periods: int = 2,
    period_duration_s: float = 240.0,
    frame_rate: float = 25.0,
    box_entry_rate: float = 0.34,
    missing_frame_rate: float = 0.0,
    pitch: Pitch = DEFAULT_PITCH,
) -> SyntheticMatch:
    """Generate a deterministic synthetic match.

    Args:
        seed: Random seed; identical seeds give byte-identical output.
        n_periods: Number of periods to simulate.
        period_duration_s: Length of each period in seconds. The default keeps
            fixtures small enough for fast tests while still yielding tens of
            positive episodes.
        frame_rate: Tracking sample rate in hertz.
        box_entry_rate: Fraction of possession sequences that reach the box.
        missing_frame_rate: Fraction of frames to blank out, for exercising the
            validator and the window-validity path.
        pitch: Pitch dimensions.

    Returns:
        The generated match with its ground-truth entry times.
    """
    rng = random.Random(seed)
    n_players = len(FORMATION_442)
    directions = _period_directions(n_periods)
    n_frames = int(period_duration_s * frame_rate)

    columns: list[_PeriodFrames] = []
    events: list[Event] = []
    true_entries: list[tuple[float, Team]] = []

    for period in range(1, n_periods + 1):
        period_clock = _PeriodClock(
            period=period,
            n_frames=n_frames,
            frame_offset=(period - 1) * n_frames,
            start_s=(period - 1) * period_duration_s,
            frame_rate=frame_rate,
        )

        # Draw order is the seed-to-output contract: sequences, then waypoints
        # per sequence, then one terminal-event draw per sequence, then the
        # missing-frame sample. The frame simulation between them draws nothing.
        sequences = _build_sequences(rng, period_duration_s, box_entry_rate)
        for seq in sequences:
            seq.waypoints = _sequence_waypoints(
                seq, rng, directions[(period, seq.team)].sign, pitch
            )

        frames = _simulate_period(sequences, directions, period_clock, pitch)
        true_entries.extend(frames.true_entries)

        for seq in sequences:
            events.extend(_sequence_events(seq, rng, period_clock))

        if missing_frame_rate > 0:
            n_missing = int(n_frames * missing_frame_rate)
            for idx in rng.sample(range(n_frames), n_missing):
                frames.ball_xy[idx] = np.nan

        columns.append(frames)

    tracking = MatchTracking(
        period=np.concatenate([c.period for c in columns]),
        frame=np.concatenate([c.frame for c in columns]),
        time_s=np.concatenate([c.time_s for c in columns]),
        home_xy=np.concatenate([c.home_xy for c in columns]),
        away_xy=np.concatenate([c.away_xy for c in columns]),
        ball_xy=np.concatenate([c.ball_xy for c in columns]),
        home_players=_player_refs(Team.HOME, n_players),
        away_players=_player_refs(Team.AWAY, n_players),
        frame_rate=frame_rate,
    )
    events.sort(key=lambda e: (e.period, e.start_frame))
    return SyntheticMatch(
        tracking=tracking,
        events=tuple(events),
        orientation=Orientation(
            directions=directions,
            report={"source": "synthetic", "seed": seed},
        ),
        true_entries=tuple(true_entries),
        frame_rate=frame_rate,
    )


def _to_unit(xy: np.ndarray, pitch: Pitch) -> np.ndarray:
    """Invert :meth:`Pitch.to_canonical` so fixtures can be written as Metrica CSV."""
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] / pitch.length + 0.5
    out[..., 1] = 0.5 - xy[..., 1] / pitch.width
    return out


def write_metrica_csv(
    match: SyntheticMatch,
    directory: Path,
    match_name: str = "Synthetic_Game",
    pitch: Pitch = DEFAULT_PITCH,
) -> dict[str, Path]:
    """Write a synthetic match in Metrica's CSV layout.

    Tests read these back through the production parser, so the fixture
    exercises the real three-row header and column pairing rather than a
    simplified stand-in.

    Args:
        match: The generated match.
        directory: Destination directory, created if absent.
        match_name: File name stem.
        pitch: Pitch dimensions used to invert the coordinate transform.

    Returns:
        Mapping of ``home``/``away``/``events`` to the written paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tracking = match.tracking
    written: dict[str, Path] = {}

    for team, label in ((Team.HOME, "Home"), (Team.AWAY, "Away")):
        players = tracking.players(team)
        unit = _to_unit(tracking.team_xy(team), pitch)
        ball_unit = _to_unit(tracking.ball_xy, pitch)
        path = directory / f"{match_name}_RawTrackingData_{label}_Team.csv"

        rows: list[str] = []
        rows.append(",,," + ",".join(f"{label}," for _ in players) + ",,")
        rows.append(",,," + ",".join(f"{p.shirt_number}," for p in players) + ",,")
        rows.append(
            "Period,Frame,Time [s],"
            + ",".join(f"Player{p.shirt_number}," for p in players)
            + "Ball,"
        )
        for i in range(tracking.n_frames):
            cells = [
                str(int(tracking.period[i])),
                str(int(tracking.frame[i])),
                f"{tracking.time_s[i]:.2f}",
            ]
            for j in range(len(players)):
                cells.extend(_fmt_pair(unit[i, j]))
            cells.extend(_fmt_pair(ball_unit[i]))
            rows.append(",".join(cells))
        path.write_text("\n".join(rows) + "\n")
        written["home" if team is Team.HOME else "away"] = path

    events_path = directory / f"{match_name}_RawEventsData.csv"
    lines = [
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
        "From,To,Start X,Start Y,End X,End Y"
    ]
    for ev in match.events:
        start = _unit_pair(ev.start_xy, pitch)
        end = _unit_pair(ev.end_xy, pitch)
        lines.append(
            ",".join(
                [
                    "Home" if ev.team is Team.HOME else "Away",
                    ev.raw_type or ev.type.value.upper().replace("_", " "),
                    ev.subtype or "",
                    str(ev.period),
                    str(ev.start_frame),
                    f"{ev.start_time_s:.2f}",
                    str(ev.end_frame),
                    f"{ev.end_time_s:.2f}",
                    ev.from_player or "",
                    ev.to_player or "",
                    *start,
                    *end,
                ]
            )
        )
    events_path.write_text("\n".join(lines) + "\n")
    written["events"] = events_path
    return written


def _fmt_pair(xy: np.ndarray) -> list[str]:
    """Format one coordinate pair, writing ``NaN`` exactly as Metrica does."""
    return ["NaN" if not np.isfinite(v) else f"{v:.5f}" for v in (xy[0], xy[1])]


def _unit_pair(xy: Sequence[float] | None, pitch: Pitch) -> list[str]:
    """Format an event coordinate pair in source units."""
    if xy is None:
        return ["NaN", "NaN"]
    arr = _to_unit(np.asarray(xy, dtype=np.float64), pitch)
    return [f"{arr[0]:.5f}", f"{arr[1]:.5f}"]
