"""Deterministic synthetic match generator.

The Metrica sample data carries no formal licence and is far too large to
commit, so every automated test runs against synthetic matches produced here.
They are generated from a seed, so fixtures are reproducible without being
stored as binary blobs.

This module is the public face of the generator: it drives
:mod:`football_insights.data.simulation` period by period and assembles what it
produces into the canonical ``MatchTracking``/``Event``/``Orientation`` the
parsers return, so nothing downstream can tell a generated match from a parsed
one. The football itself — possession, touches, passes, interceptions, shots,
clearances and the ball's trajectory — lives there.

What the simulation guarantees, and why it matters here: the ball is ballistic
between touches, and its velocity is written in exactly one place, which
records an event every time. So the ball cannot change direction without a
recorded cause, and possession is held by whoever last reached it rather than
by a schedule. The previous generator interpolated the ball along waypoints,
which curved it continuously with nothing causing the curve.

Motion is calibrated against aggregate distributions measured from the Metrica
sample — speed, acceleration, spacing, possession duration, pass length and
flight time. No coordinate sequence, event timeline or identifier from that
dataset is copied into anything generated here; see
``scripts/calibrate-from-metrica.py``.

It also writes real Metrica-format CSV, which lets the production parser be
tested against the exact on-disk layout it will meet in the wild.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from football_insights.data.profiles import DEFAULT_PROFILE, FixtureProfile
from football_insights.data.simulation import N_PLAYERS, SimulatedPeriod, simulate_period
from football_insights.domain import (
    AttackDirection,
    Event,
    MatchTracking,
    Orientation,
    PlayerRef,
    Team,
)
from football_insights.pitch import DEFAULT_PITCH, Pitch

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    profile: FixtureProfile = DEFAULT_PROFILE,
    missing_frame_rate: float = 0.0,
    pitch: Pitch = DEFAULT_PITCH,
) -> SyntheticMatch:
    """Generate a deterministic synthetic match.

    Every frame comes from :mod:`football_insights.data.simulation`, which
    resolves possession, touches and the ball's trajectory from the state of
    the players rather than from a schedule. The seed fixes the whole run: the
    simulation draws from one generator in a deterministic order, so identical
    arguments give identical output.

    Args:
        seed: Random seed; identical seeds give identical output.
        n_periods: Number of periods to simulate.
        period_duration_s: Length of each period in seconds.
        frame_rate: Tracking sample rate in hertz.
        profile: Tactical archetype; see :mod:`football_insights.data.profiles`.
        missing_frame_rate: Fraction of frames to blank the ball in, for
            exercising the validator and the window-validity path.
        pitch: Pitch dimensions.

    Returns:
        The generated match with its ground-truth entry times.
    """
    rng = random.Random(seed)
    directions = _period_directions(n_periods)
    n_frames = int(period_duration_s * frame_rate)

    periods: list[SimulatedPeriod] = []
    events: list[Event] = []
    true_entries: list[tuple[float, Team]] = []

    for period in range(1, n_periods + 1):
        simulated = simulate_period(
            rng,
            profile,
            period=period,
            n_frames=n_frames,
            frame_offset=(period - 1) * n_frames,
            start_s=(period - 1) * period_duration_s,
            frame_rate=frame_rate,
            attack_sign={team: directions[(period, team)].sign for team in (Team.HOME, Team.AWAY)},
            pitch=pitch,
        )
        events.extend(simulated.events)
        true_entries.extend(simulated.true_entries)

        if missing_frame_rate > 0:
            n_missing = int(n_frames * missing_frame_rate)
            for idx in rng.sample(range(n_frames), n_missing):
                simulated.ball_xy[idx] = np.nan

        periods.append(simulated)

    frames_per_period = np.arange(1, n_frames + 1, dtype=np.int64)
    tracking = MatchTracking(
        period=np.concatenate(
            [np.full(n_frames, p, dtype=np.int16) for p in range(1, n_periods + 1)]
        ),
        frame=np.concatenate(
            [frames_per_period + (p - 1) * n_frames for p in range(1, n_periods + 1)]
        ),
        time_s=np.concatenate(
            [
                (p - 1) * period_duration_s + frames_per_period / frame_rate
                for p in range(1, n_periods + 1)
            ]
        ),
        home_xy=np.concatenate([s.home_xy for s in periods]),
        away_xy=np.concatenate([s.away_xy for s in periods]),
        ball_xy=np.concatenate([s.ball_xy for s in periods]),
        home_players=_player_refs(Team.HOME, N_PLAYERS),
        away_players=_player_refs(Team.AWAY, N_PLAYERS),
        frame_rate=frame_rate,
    )
    events.sort(key=lambda e: (e.period, e.start_frame))
    return SyntheticMatch(
        tracking=tracking,
        events=tuple(events),
        orientation=Orientation(
            directions=directions,
            report={"source": "synthetic", "seed": seed, "profile": profile.key},
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
