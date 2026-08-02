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

Motion, and what it is not
--------------------------
Players carry velocity and are bound by acceleration, deceleration and
turn-rate limits; each has their own top speed, reaction delay and smoothed
view of where play is. That replaced a direct step toward a shared formation
target, which produced movement that was 93.5% collinear frame to frame, had a
median acceleration of exactly zero, and correlated team-mate directions at
0.99 — a team translating as one rigid object.

Measured against a Metrica sample match, the guardrails now hold with margin:
collinearity 0.28 against 0.28, team-mate correlation below 0.67 against 0.48,
essentially nothing at the speed cap against 0.2%, and heading changes at 1.1
degrees a frame against 1.0.

Two statistics remain some way off, and honestly so. Median speed is about
4.5 m/s against 1.5, and median acceleration about 4 m/s² against 1.7. Both
have the same cause: this fixture is continuous end-to-end football. There are
no throw-ins, substitutions, injuries, walking phases or recovery jogs, so the
12% of real frames spent below 0.5 m/s have no counterpart here. That is a
property of the *fixture design*, not of the motion model, and closing it would
mean simulating stoppages rather than retuning any constant below.
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

#: Which formation slots are goalkeeper, defender, midfielder and forward.
#: Indices into :data:`FORMATION_442`.
_ROLE_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("gk", 0, 1),
    ("def", 1, 5),
    ("mid", 5, 9),
    ("fwd", 9, 11),
)

#: Top speed and acceleration by role, before per-player variation. Forwards are
#: quickest; a goalkeeper is not.
#:
#: Top speeds sit near the 99th percentile of the Metrica sample, 6.1 m/s, and
#: are chosen so the fastest role in the fastest profile with the luckiest draw
#: still lands under 7.4 m/s. The 8.0 m/s cap is a bound the data must never
#: cross, not a speed anyone should travel at: a top speed just beneath it means
#: the quickest players spend part of every match pinned to it, which is what
#: put 12% of frames at exactly 8.00 m/s before.
_ROLE_SPEED_MS: Final[dict[str, float]] = {"gk": 4.8, "def": 6.1, "mid": 6.3, "fwd": 6.6}
_ROLE_ACCEL_MS2: Final[dict[str, float]] = {"gk": 3.2, "def": 4.4, "mid": 4.6, "fwd": 5.0}

#: Deceleration is faster than acceleration — stopping is easier than starting.
_DECEL_RATIO: Final = 1.35

#: Turning is speed-dependent: a standing player can pivot, a sprinting one
#: cannot. Yaw rate is ``_TURN_BASE_RAD_S / (1 + speed / _TURN_REF_MS)``.
_TURN_BASE_RAD_S: Final = 7.0
_TURN_REF_MS: Final = 2.6

#: Below this a player is treated as stationary and may set off in any
#: direction, so the turn limit never traps someone who has stopped.
_TURN_FREE_SPEED_MS: Final = 0.4

#: Seconds a player takes to arrive at a target from full speed. Larger values
#: make the approach lazier; this is what stops the formation snapping.
_ARRIVAL_TIME_S: Final = 0.30

#: Reaction delay range in frames at 25 Hz - 80 to 480 ms. Part of what stops
#: eleven players responding to the same ball position in the same frame.
_REACTION_FRAMES: Final[tuple[int, int]] = (2, 12)

#: Time constant, in seconds, of each player's *view* of where play is.
#:
#: Team shape follows the run of play, not the ball. A pass travels at 15 m/s;
#: a defensive line does not, and an earlier version that pointed the formation
#: target straight at the current ball position had every outfield player
#: chasing it at a median 4 m/s where the Metrica sample sits at 1.5, with
#: team-mate directions correlating at 0.85.
#:
#: Smoothing per player rather than globally is what makes the shape deform
#: instead of translating: eleven different lags mean eleven different ideas of
#: where the ball is, which is also closer to the truth than one shared one.
_ROLE_FOLLOW_TAU_S: Final[dict[str, float]] = {"gk": 2.6, "def": 1.1, "mid": 1.5, "fwd": 2.1}

#: Off-ball movement: a per-player wander added to the formation target, as two
#: harmonics per axis so the path is not a recognisable sine.
#:
#: Amplitude and frequency pull in different directions, which is why they were
#: tuned separately: amplitude is what de-synchronises a team, and frequency
#: squared is what produces acceleration.
#:
#: A player tracking a slowly moving target reaches top speed and stays there,
#: which is constant velocity and no acceleration at all. Turning the wander
#: down far enough to match the reference's median speed instead puts the shared
#: ball-following back in charge and sends team-mate correlation above 0.85, so
#: the two cannot be satisfied by the same setting. These values keep every
#: guardrail while leaving the motion busier than the reference; see the module
#: docstring for what that costs.
_OFF_BALL_HARMONICS: Final = 2
_OFF_BALL_AMPLITUDE_M: Final[tuple[float, float]] = (2.0, 5.2)
_OFF_BALL_FREQ_HZ: Final[tuple[float, float]] = (0.05, 0.13)

#: Ball acceleration and drag. The cap becomes a limit the ball reaches
#: occasionally rather than the speed it travels at.
_BALL_ACCEL_MS2: Final = 55.0
_BALL_DRAG_PER_S: Final = 0.85
_BALL_ARRIVAL_S: Final = 0.30

#: Speed a hard pass travels at, before the profile's scale.
#:
#: Deliberately below :data:`MAX_BALL_SPEED_MS`, and scaled from here rather
#: than from the cap. Scaling the cap put the quickest profile's target at
#: 25.5 m/s, so the clamp became the operating point and the 99th percentile of
#: ball speed sat at exactly 22.00 again — the same defect the velocity model
#: was added to remove, reintroduced through a multiplier.
_NOMINAL_BALL_SPEED_MS: Final = 17.0


@dataclass(frozen=True, slots=True)
class FixtureProfile:
    """A tactical archetype, as parameters rather than as a separate generator.

    One simulation drives all three fixtures. Three implementations would drift
    apart the first time anything downstream changed, and there would be no
    honest way to say the fixtures differ in *tactics* rather than in code.

    Attributes:
        key: Stable identifier, used in match ids and metadata.
        name: Short human name for the UI.
        narrative: One line describing what a viewer should expect to see.
        sequence_duration_s: Range a possession sequence is drawn from.
        dead_ball_gap_s: Range of the pause between sequences.
        box_entry_rate: Fraction of sequences that reach the penalty area.
        max_sequences_between_entries: Longest run of sequences allowed to pass
            without one reaching the box. Bounds the quiet stretches: drawing
            each sequence independently lets several barren ones fall together
            by chance, and a viewer who arrives during that run sees a
            demonstration that appears to do nothing.
        turnover_prob: Chance possession changes hands between sequences.
        leg_duration_s: Seconds per waypoint leg; shorter means more passes.
        width_scale: Multiplies the formation's lateral spread.
        line_offset_m: Shifts the defending block up (+) or back (-) the pitch.
        press_squeeze: How hard the defending side collapses toward the ball.
        defend_reaction_scale: Multiplies the defending team's reaction delay.
            Below 1 is an aggressive press; above 1 is a side that sits off.
        progression: Shape of the ball's advance through a sequence.
        ball_speed_scale: Multiplies the ball's target speed.
        player_speed_scale: Multiplies every player's top speed.
        accel_scale: Multiplies every player's acceleration.
    """

    key: str
    name: str
    narrative: str
    sequence_duration_s: tuple[float, float]
    dead_ball_gap_s: tuple[float, float]
    box_entry_rate: float
    max_sequences_between_entries: int
    turnover_prob: float
    leg_duration_s: float
    width_scale: float
    line_offset_m: float
    press_squeeze: float
    defend_reaction_scale: float
    progression: str
    ball_speed_scale: float
    player_speed_scale: float
    accel_scale: float


#: The three public archetypes, in rotation order.
#:
#: They are meant to be told apart by watching, not by reading a label: a
#: patient side that recycles possession, a pressing side that wins the ball
#: high and loses it quickly, and a deep block that breaks at speed. The
#: parameters below are what produce that, and the motion statistics they
#: produce are asserted against in the test suite.
PROFILES: Final[tuple[FixtureProfile, ...]] = (
    FixtureProfile(
        key="build_up",
        name="Patient build-up",
        narrative="Long possessions recycled through midfield, with a wide shape.",
        sequence_duration_s=(13.0, 26.0),
        dead_ball_gap_s=(1.0, 2.5),
        box_entry_rate=0.46,
        max_sequences_between_entries=2,
        turnover_prob=0.55,
        leg_duration_s=3.2,
        width_scale=1.18,
        line_offset_m=-1.0,
        press_squeeze=0.30,
        defend_reaction_scale=1.35,
        progression="ease",
        ball_speed_scale=0.85,
        player_speed_scale=0.94,
        accel_scale=0.90,
    ),
    FixtureProfile(
        key="high_press",
        name="High press and turnovers",
        narrative="Compressed shape, quick turnovers and short possessions.",
        sequence_duration_s=(4.5, 11.0),
        dead_ball_gap_s=(0.8, 2.0),
        box_entry_rate=0.48,
        max_sequences_between_entries=3,
        turnover_prob=0.90,
        leg_duration_s=2.0,
        width_scale=0.82,
        line_offset_m=7.0,
        press_squeeze=0.56,
        defend_reaction_scale=0.60,
        progression="linear",
        ball_speed_scale=1.02,
        player_speed_scale=1.02,
        accel_scale=1.12,
    ),
    FixtureProfile(
        key="counter",
        name="Fast counterattacks",
        narrative="A deep block that breaks quickly and vertically on the regain.",
        sequence_duration_s=(6.0, 15.0),
        dead_ball_gap_s=(0.8, 2.2),
        box_entry_rate=0.60,
        max_sequences_between_entries=2,
        turnover_prob=0.78,
        leg_duration_s=2.4,
        width_scale=0.96,
        line_offset_m=-7.0,
        press_squeeze=0.42,
        defend_reaction_scale=0.95,
        progression="late_burst",
        ball_speed_scale=1.16,
        player_speed_scale=1.06,
        accel_scale=1.20,
    ),
)

DEFAULT_PROFILE: Final = PROFILES[0]

PROFILES_BY_KEY: Final[dict[str, FixtureProfile]] = {p.key: p for p in PROFILES}


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


@dataclass(frozen=True, slots=True)
class _PlayerMotion:
    """Per-player movement limits and quirks, all shape ``(11,)`` or ``(11, 2)``.

    Drawn once per team per match, in a fixed order, so the seed still
    determines the output exactly. Making these per-player rather than global
    is most of what stops a team moving as one object: eleven players with the
    same top speed, the same acceleration and the same reaction time will trace
    parallel paths whatever the steering does.
    """

    max_speed: np.ndarray
    max_accel: np.ndarray
    max_decel: np.ndarray
    turn_base: np.ndarray
    #: Frames each player lags the ball by. See :class:`_BallHistory`.
    reaction: np.ndarray
    #: Seconds of smoothing on that player's view of where play is.
    follow_tau: np.ndarray
    #: Shape ``(n, 2, _OFF_BALL_HARMONICS)`` — player, axis, harmonic.
    off_ball_amp: np.ndarray
    off_ball_freq: np.ndarray
    off_ball_phase: np.ndarray

    def drift(self, now: float) -> np.ndarray:
        """Off-ball offset for every player at match time ``now``, ``(n, 2)``."""
        waves = self.off_ball_amp * np.sin(
            2.0 * math.pi * self.off_ball_freq * now + self.off_ball_phase
        )
        return np.asarray(waves.sum(axis=2))


def _draw_player_motion(
    rng: random.Random, n_players: int, profile: FixtureProfile, *, defending: bool
) -> _PlayerMotion:
    """Draw one team's movement constants.

    The draw order is part of the seed-to-output contract, exactly as the
    sequence and waypoint draws are: role bands in order, and within each band
    every player's speed, acceleration, turn rate, reaction and off-ball terms
    in that order. Reordering any of it changes every fixture.

    Args:
        rng: Seeded source; consumed in a fixed order.
        n_players: Squad size, matching :data:`FORMATION_442`.
        profile: Supplies the speed, acceleration and pressing scales.
        defending: Whether this team starts the match without the ball. Only
            the reaction scale differs, and it is what makes a pressing side
            feel like one.

    Returns:
        The team's constants.
    """
    roles = ["mid"] * n_players
    for name, start, stop in _ROLE_BANDS:
        for index in range(start, min(stop, n_players)):
            roles[index] = name

    reaction_scale = profile.defend_reaction_scale if defending else 1.0
    speed = np.empty(n_players)
    accel = np.empty(n_players)
    turn = np.empty(n_players)
    reaction = np.empty(n_players, dtype=np.int64)
    tau = np.empty(n_players)
    amp = np.empty((n_players, 2, _OFF_BALL_HARMONICS))
    freq = np.empty((n_players, 2, _OFF_BALL_HARMONICS))
    phase = np.empty((n_players, 2, _OFF_BALL_HARMONICS))

    lo_react, hi_react = _REACTION_FRAMES
    lo_amp, hi_amp = _OFF_BALL_AMPLITUDE_M
    lo_freq, hi_freq = _OFF_BALL_FREQ_HZ

    for i in range(n_players):
        role = roles[i]
        speed[i] = _ROLE_SPEED_MS[role] * profile.player_speed_scale * rng.uniform(0.94, 1.06)
        accel[i] = _ROLE_ACCEL_MS2[role] * profile.accel_scale * rng.uniform(0.90, 1.10)
        turn[i] = _TURN_BASE_RAD_S * rng.uniform(0.85, 1.15)
        reaction[i] = round(rng.uniform(lo_react, hi_react) * reaction_scale)
        # A pressing side reads play sooner as well as reacting faster, so the
        # same scale drives both. Its spread is wide on purpose: a shared time
        # constant would put the whole team back on one trajectory.
        tau[i] = _ROLE_FOLLOW_TAU_S[role] * reaction_scale * rng.uniform(0.55, 1.75)
        # The keeper does not roam; everyone else works off the ball.
        scale = 0.25 if role == "gk" else 1.0
        for axis in range(2):
            # Longitudinal movement is smaller than lateral: a shape stretches
            # sideways far more readily than it breaks its own line.
            axis_scale = 0.65 if axis == 0 else 1.0
            for harmonic in range(_OFF_BALL_HARMONICS):
                # The second harmonic is smaller and faster, which turns a
                # recognisable sine into something closer to a run-and-check.
                taper = 1.0 if harmonic == 0 else 0.55
                amp[i, axis, harmonic] = rng.uniform(lo_amp, hi_amp) * scale * axis_scale * taper
                freq[i, axis, harmonic] = rng.uniform(lo_freq, hi_freq) * (1.0 + 1.7 * harmonic)
                phase[i, axis, harmonic] = rng.uniform(0.0, 2.0 * math.pi)

    np.clip(speed, 0.5, MAX_PLAYER_SPEED_MS, out=speed)
    reaction = np.clip(reaction, 0, _REACTION_FRAMES[1] * 2)
    return _PlayerMotion(
        max_speed=speed,
        max_accel=accel,
        max_decel=accel * _DECEL_RATIO,
        turn_base=turn,
        reaction=reaction,
        follow_tau=np.clip(tau, 0.2, 6.0),
        off_ball_amp=amp,
        off_ball_freq=freq,
        off_ball_phase=phase,
    )


class _BallHistory:
    """Recent ball positions, so each player can react to a different one.

    A ring buffer rather than a list because it is read every frame by every
    player and its length is fixed by the largest reaction delay. Reading a
    per-player delay out of it is one fancy-index, which keeps the whole
    simulation vectorised.
    """

    __slots__ = ("_buffer", "_head")

    def __init__(self, capacity: int, initial: np.ndarray) -> None:
        """Fill the history with the starting position, so frame 0 is defined."""
        self._buffer = np.tile(np.asarray(initial, dtype=np.float64), (capacity, 1))
        self._head = 0

    def push(self, xy: np.ndarray) -> None:
        """Record this frame's ball position."""
        self._head = (self._head + 1) % len(self._buffer)
        self._buffer[self._head] = xy

    def delayed(self, frames: np.ndarray) -> np.ndarray:
        """Ball position each player currently believes, shape ``(n, 2)``."""
        return np.asarray(self._buffer[(self._head - frames) % len(self._buffer)])


def _formation_targets(
    ball_per_player: np.ndarray,
    attacking: bool,
    direction_sign: float,
    pitch: Pitch,
    profile: FixtureProfile,
    off_ball: np.ndarray,
) -> np.ndarray:
    """Target positions for one team, one ball position per player.

    The block still slides up and down the pitch with the ball and squeezes
    toward it laterally — that is what makes compactness and line-height
    features carry signal. What changed is that every player reads a *different*
    ball position, delayed by their own reaction time, and adds their own slow
    off-ball drift. Previously this function took a single ball position and
    applied one shift and one squeeze to all ten outfield players, so the ten
    targets were a rigid translation of each other and the team moved as one
    object: measured against real tracking, team-mate velocity direction
    correlated at 0.99 where Metrica sits near 0.48.

    Args:
        ball_per_player: Ball position each player is reacting to, ``(n, 2)``.
        attacking: Whether this team is in possession.
        direction_sign: ``+1`` if this team attacks ``+x``, ``-1`` otherwise.
        pitch: Pitch dimensions.
        profile: Supplies width, line height and pressing intensity.
        off_ball: Per-player drift to add, ``(n, 2)``.

    Returns:
        Array of shape ``(n, 2)`` in canonical metres.
    """
    template = np.asarray(FORMATION_442, dtype=np.float64)
    xy = np.empty_like(template)
    xy[:, 0] = template[:, 0] * pitch.half_length
    xy[:, 1] = template[:, 1] * pitch.half_width * profile.width_scale

    # Orient the template to this team's attacking direction.
    xy *= direction_sign

    ball_progress = ball_per_player[:, 0] * direction_sign  # +ve when upfield

    # Attackers push higher; defenders hold a line relative to the ball.
    if attacking:
        shift = 0.55 * ball_progress
    else:
        shift = 0.45 * ball_progress + 8.0 + profile.line_offset_m
    xy[1:, 0] += shift[1:] * direction_sign

    squeeze = 0.25 if attacking else profile.press_squeeze
    xy[1:, 1] = xy[1:, 1] * (1.0 - squeeze) + ball_per_player[1:, 1] * squeeze

    xy += off_ball

    # Goalkeeper stays home, tracking the ball's lateral position a little.
    xy[0, 0] = -0.96 * pitch.half_length * direction_sign
    xy[0, 1] = 0.22 * ball_per_player[0, 1]

    np.clip(xy[:, 0], -pitch.half_length + 1.0, pitch.half_length - 1.0, out=xy[:, 0])
    np.clip(xy[:, 1], -pitch.half_width + 1.0, pitch.half_width - 1.0, out=xy[:, 1])
    return xy


def _steer(
    pos: np.ndarray, vel: np.ndarray, target: np.ndarray, motion: _PlayerMotion, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one team a frame under acceleration and turning limits.

    Replaces a direct step toward the target capped at a per-frame distance.
    That version had no state at all: the position moved straight at whatever
    the target was, so speed was either zero or exactly the cap and direction
    changed only when the target did. Measured against Metrica, 93.5% of
    three-frame segments were collinear to within half a degree and the median
    acceleration was exactly zero.

    Three limits, applied in order — desired velocity, then how fast it may be
    reached, then how sharply the heading may change:

    * a desired velocity toward the target, easing off over the last stride so
      arriving does not mean stopping dead;
    * an acceleration cap, with a larger one for slowing down than speeding up;
    * a yaw-rate cap that tightens with speed, so a sprinting player arcs where
      a walking one can pivot.

    Args:
        pos: Current positions, ``(n, 2)``.
        vel: Current velocities, ``(n, 2)`` in m/s.
        target: Where each player is trying to be, ``(n, 2)``.
        motion: This team's limits.
        dt: Seconds per frame.

    Returns:
        The new positions and velocities.
    """
    delta = target - pos
    dist = np.hypot(delta[:, 0], delta[:, 1])
    safe = np.maximum(dist, 1e-9)

    # Ease off over the last stride rather than arriving at full pace and
    # stopping, which is what makes a formation look like it is snapping.
    wanted_speed = np.minimum(motion.max_speed, dist / _ARRIVAL_TIME_S)
    desired = delta / safe[:, None] * wanted_speed[:, None]

    dv = desired - vel
    speeding_up = np.hypot(desired[:, 0], desired[:, 1]) >= np.hypot(vel[:, 0], vel[:, 1])
    limit = np.where(speeding_up, motion.max_accel, motion.max_decel) * dt
    dv_mag = np.hypot(dv[:, 0], dv[:, 1])
    scale = np.where(dv_mag > limit, limit / np.maximum(dv_mag, 1e-9), 1.0)
    candidate = vel + dv * scale[:, None]

    # Yaw-rate limit. A player who has effectively stopped may set off in any
    # direction; one at pace may not simply reverse.
    speed_now = np.hypot(vel[:, 0], vel[:, 1])
    speed_next = np.hypot(candidate[:, 0], candidate[:, 1])
    max_turn = motion.turn_base / (1.0 + speed_now / _TURN_REF_MS) * dt
    heading_now = np.arctan2(vel[:, 1], vel[:, 0])
    heading_next = np.arctan2(candidate[:, 1], candidate[:, 0])
    swing = np.arctan2(np.sin(heading_next - heading_now), np.cos(heading_next - heading_now))
    limited = np.clip(swing, -max_turn, max_turn)
    free = (speed_now < _TURN_FREE_SPEED_MS) | (speed_next < 1e-9)
    heading = np.where(free, heading_next, heading_now + limited)

    new_vel = np.stack([speed_next * np.cos(heading), speed_next * np.sin(heading)], axis=1)
    return pos + new_vel * dt, new_vel


def _build_sequences(
    rng: random.Random,
    period_duration_s: float,
    profile: FixtureProfile,
) -> list[_Sequence]:
    """Lay out possession sequences across one period.

    Sequence length, how often possession turns over and how often a move
    reaches the box are the profile's, not constants: they are most of what
    separates a side that recycles the ball from one that wins it high and
    gives it straight back.
    """
    sequences: list[_Sequence] = []
    t = 0.0
    team = Team.HOME
    lo, hi = profile.sequence_duration_s
    gap_lo, gap_hi = profile.dead_ball_gap_s
    barren = 0
    # A tail shorter than the longest sequence would be cut off mid-move.
    while t < period_duration_s - hi * 0.6:
        duration = rng.uniform(lo, hi)
        # The draw happens either way, so forcing an entry never shifts the
        # seed-to-output mapping — it only overrides the result.
        reaches_box = rng.random() < profile.box_entry_rate
        if barren >= profile.max_sequences_between_entries:
            reaches_box = True
        barren = 0 if reaches_box else barren + 1
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
        t += duration + rng.uniform(gap_lo, gap_hi)
        if rng.random() < profile.turnover_prob:
            team = team.opponent
    return sequences


def _progress(fraction: float, shape: str) -> float:
    """How far up the pitch a move has got, as a function of how far through it is.

    The three shapes are the difference between a side that works the ball
    forward steadily, one that goes direct, and one that sits and then breaks.

    Args:
        fraction: Position through the sequence, 0 to 1.
        shape: ``"ease"``, ``"linear"`` or ``"late_burst"``.

    Returns:
        Progress upfield, 0 to 1.
    """
    if shape == "linear":
        return fraction
    if shape == "late_burst":
        # Nearly static, then a sharp vertical break in the last third.
        return float(fraction**2.6)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _sequence_waypoints(
    seq: _Sequence,
    rng: random.Random,
    direction_sign: float,
    pitch: Pitch,
    profile: FixtureProfile,
) -> list[tuple[float, float]]:
    """Ball waypoints from deep possession to either a box entry or a breakdown."""
    n_legs = max(2, int(seq.duration_s / profile.leg_duration_s))
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
        eased = _progress(f, profile.progression)
        x = start_x + (end_x - start_x) * eased
        y = start_y + (end_y - start_y) * eased
        if 0 < i < n_legs:
            x += rng.uniform(-4.0, 4.0)
            # A patient side switches play more; a counter goes straight.
            y += rng.uniform(-9.0, 9.0) * profile.width_scale
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


def _advance_ball(
    ball: FloatArray,
    vel: FloatArray,
    active: _Sequence,
    t_rel: float,
    dt: float,
    profile: FixtureProfile,
) -> tuple[FloatArray, FloatArray]:
    """Move the ball one frame along the active sequence's waypoints.

    The ball carries velocity and sheds it to drag, rather than being teleported
    toward its target at whatever speed the gap implies and clipped at the cap.
    The clipped version reached the cap constantly: the 99th percentile of ball
    speed was 22.00 m/s, exactly the limit, where real tracking reaches its own
    99th percentile at a speed nothing imposed.

    Args:
        ball: Current position.
        vel: Current velocity in m/s.
        active: The possession sequence being played out.
        t_rel: Seconds into the period.
        dt: Seconds per frame.
        profile: Supplies the ball speed scale.

    Returns:
        The new position and velocity.
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
    ceiling = _NOMINAL_BALL_SPEED_MS * profile.ball_speed_scale
    wanted = min(ceiling, dist / _BALL_ARRIVAL_S)
    desired = delta / max(dist, 1e-9) * wanted

    dv = desired - vel
    dv_mag = float(np.hypot(*dv))
    cap = _BALL_ACCEL_MS2 * dt
    new_vel: FloatArray = vel + dv * (cap / dv_mag if dv_mag > cap else 1.0)
    new_vel = new_vel * (1.0 - _BALL_DRAG_PER_S * dt)

    speed = float(np.hypot(*new_vel))
    if speed > MAX_BALL_SPEED_MS:
        new_vel = new_vel * (MAX_BALL_SPEED_MS / speed)
    moved: FloatArray = ball + new_vel * dt
    return moved, new_vel


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
    profile: FixtureProfile,
    motions: dict[Team, _PlayerMotion],
) -> _PeriodFrames:
    """Step the ball and both teams through one period, frame by frame.

    Draws no random numbers: the randomness is entirely in the sequences,
    waypoints and per-player constants handed in, which keeps the
    seed-to-output mapping easy to reason about.
    """
    period, n_frames, dt = clock.period, clock.n_frames, clock.dt
    n_players = len(FORMATION_442)
    zero_off = np.zeros((n_players, 2))
    home_sign = directions[(period, Team.HOME)].sign
    away_sign = directions[(period, Team.AWAY)].sign
    home_pos = _formation_targets(
        np.zeros((n_players, 2)), True, home_sign, pitch, profile, zero_off
    )
    away_pos = _formation_targets(
        np.zeros((n_players, 2)), False, away_sign, pitch, profile, zero_off
    )
    home_vel = np.zeros((n_players, 2))
    away_vel = np.zeros((n_players, 2))

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
    ball_vel = np.zeros(2)
    in_box_prev = False
    longest_reaction = int(
        max(motions[Team.HOME].reaction.max(), motions[Team.AWAY].reaction.max())
    )
    history = _BallHistory(longest_reaction + 2, ball)
    # Each player's own smoothed idea of where play is, and the per-frame weight
    # that pulls it toward what they can currently see. Started on the centre
    # spot, which is where the ball is on the first frame.
    views = {team: np.zeros((n_players, 2)) for team in (Team.HOME, Team.AWAY)}
    alphas = {
        team: (1.0 - np.exp(-dt / motions[team].follow_tau))[:, None]
        for team in (Team.HOME, Team.AWAY)
    }

    for i in range(n_frames):
        t_rel = (i + 1) * dt
        seq_index, active = _sequence_at(sequences, seq_index, t_rel)

        if active is None:
            # Dead ball: the ball rolls to a stop rather than being scaled
            # toward the centre, which is not a motion any ball performs.
            ball_vel = ball_vel * (1.0 - 2.5 * dt)
            ball = ball + ball_vel * dt
            attacking_team = Team.HOME
            in_possession = False
        else:
            attacking_team = active.team
            in_possession = True
            ball, ball_vel = _advance_ball(ball, ball_vel, active, t_rel, dt, profile)

        history.push(ball)
        # Absolute match time, so a player's off-ball phase carries across the
        # half-time break instead of resetting every period.
        now = clock.start_s + t_rel

        for team, sign, attacking in (
            (Team.HOME, home_sign, attacking_team is Team.HOME and in_possession),
            (Team.AWAY, away_sign, attacking_team is Team.AWAY and in_possession),
        ):
            motion = motions[team]
            # Delay, then smooth: what each player saw a moment ago, folded
            # into where they already believed play was.
            seen = history.delayed(motion.reaction)
            view = views[team]
            view += alphas[team] * (seen - view)
            target = _formation_targets(view, attacking, sign, pitch, profile, motion.drift(now))
            if team is Team.HOME:
                home_pos, home_vel = _steer(home_pos, home_vel, target, motion, dt)
            else:
                away_pos, away_vel = _steer(away_pos, away_vel, target, motion, dt)

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
    profile: FixtureProfile = DEFAULT_PROFILE,
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
        profile: Tactical archetype. Decides possession length, team shape,
            pressing intensity, progression and the movement scales, so two
            fixtures on the same seed with different profiles are different
            matches rather than the same one relabelled.
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

    # Drawn before any period, so the same squad plays both halves. Home first,
    # then away: part of the seed-to-output contract like every other draw here.
    motions = {
        Team.HOME: _draw_player_motion(rng, n_players, profile, defending=False),
        Team.AWAY: _draw_player_motion(rng, n_players, profile, defending=True),
    }

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

        # Draw order is the seed-to-output contract: player constants above,
        # then per period sequences, waypoints per sequence, one terminal-event
        # draw per sequence, and the missing-frame sample. The frame simulation
        # between them draws nothing.
        sequences = _build_sequences(rng, period_duration_s, profile)
        for seq in sequences:
            seq.waypoints = _sequence_waypoints(
                seq, rng, directions[(period, seq.team)].sign, pitch, profile
            )

        frames = _simulate_period(sequences, directions, period_clock, pitch, profile, motions)
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
