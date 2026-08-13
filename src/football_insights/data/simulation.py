"""Event-driven football simulation.

The ball is ballistic at all times: it travels in a straight line and loses
speed to drag. Its velocity is written in exactly one place — :meth:`_Sim._touch`
— and every call to that method records an event. A boundary crossing is the
only other thing that ends a trajectory, and it too is recorded.

That is the whole design, and it is what makes the central invariant structural
rather than something a test hopes to catch: **the ball cannot change direction
without a recorded cause**, because there is no code path that moves it other
than a straight step, a touch, or going out of play.

Everything else falls out of it:

* a *dribble* is a player pushing the ball two or three metres ahead, chasing
  it down and pushing again — a sequence of recorded touches, with the ball
  rolling straight between them;
* an *interception* is a defender happening to be the nearest player when the
  ball arrives, which needs no separate mechanism;
* a *deflection* is a defender close to the ball's path who redirects it
  without controlling it;
* *possession* is whoever touched it last, so a player in possession is by
  construction within a touch radius of the ball.

The previous generator moved the ball along interpolated waypoints, so it
curved continuously with nothing causing the curve, and possession was a
property of a schedule rather than of anyone being near the ball.

Calibration
-----------
The constants below are calibrated against aggregate distributions measured
from the Metrica sample matches — speed, acceleration, spacing, possession
duration, pass length and flight time — and nothing else. No coordinate
sequence, event timeline or identifier from that dataset appears here or in
anything this module produces. ``scripts/calibrate-from-metrica.py``
recomputes the distributions so the numbers can be checked against their
source.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Final

import numpy as np

from football_insights.domain import Event, EventType, Team
from football_insights.pitch import DEFAULT_PITCH, Pitch

if TYPE_CHECKING:
    from football_insights.data.profiles import FixtureProfile

# ---------------------------------------------------------------- calibration
#
# Percentiles measured across the three Metrica sample matches. Each constant
# names the statistic it was fitted to, so a reader can check the simulation
# against the reference rather than against taste.

#: Player speed: reference p50 1.50, p90 3.83, p99 6.19 m/s.
MAX_PLAYER_SPEED_MS: Final = 8.0
_ROLE_SPEED_MS: Final[dict[str, float]] = {"gk": 4.8, "def": 6.1, "mid": 6.3, "fwd": 6.6}

#: Player acceleration: reference p50 1.70, p90 4.30, p99 10.04 m/s^2.
_ROLE_ACCEL_MS2: Final[dict[str, float]] = {"gk": 3.2, "def": 4.4, "mid": 4.6, "fwd": 5.0}
_DECEL_RATIO: Final = 1.35

#: Turning is speed-dependent: a standing player pivots, a sprinting one arcs.
_TURN_BASE_RAD_S: Final = 7.0
_TURN_REF_MS: Final = 2.6
_TURN_FREE_SPEED_MS: Final = 0.4

#: Ball speed: reference p50 5.82, p75 11.25, p90 15.29, p99 22.50 m/s.
MAX_BALL_SPEED_MS: Final = 22.0
#: Rolling resistance, tuned so a 14 m pass — the reference median length —
#: takes about 1.2 s, the reference median flight time.
_BALL_DRAG_PER_S: Final = 0.62

#: Pass length: reference p25 9.75, p50 14.09, p75 20.33, p90 27.82 m.
_PASS_LENGTH_M: Final[tuple[float, float]] = (8.0, 28.0)
#: A clearance is not a pass; it is distance for its own sake.
_CLEARANCE_LENGTH_M: Final[tuple[float, float]] = (25.0, 45.0)

#: Nearest team-mate: reference p25 6.73, p50 9.65, p90 16.11 m. Used as the
#: spacing the support runs aim to preserve.
_SUPPORT_SPACING_M: Final = 9.5

#: How close a player must be to put a foot on the ball.
#:
#: Public because it is part of the contract, not an implementation detail: the
#: possession-proximity invariant is defined in terms of it, so the test
#: asserting it has to read the same number the simulation enforces. Two metres is a stride
#: and a reach; below about 1.5 the ball squirts through crowds untouched, and
#: above 3 possession changes hands from across the pitch.
TOUCH_RADIUS_M: Final = 2.0

#: A defender this close to a moving ball may get something on it without
#: controlling it.
_DEFLECT_RADIUS_M: Final = 3.2

#: Chance per *second* of flight that an eligible defender gets a touch.
#:
#: A rate rather than a per-frame probability, and the distinction is the whole
#: point. The same 0.05 tested once per frame is 1.25 deflections per second at
#: 25 Hz, which gave a typical 1.5 s pass an 85% chance of being deflected and
#: put the count at 522-666 per 90 against a reference in the tens. Multiplied
#: by the timestep it means what it reads as: about 7% of passes.
_DEFLECT_RATE_PER_S: Final = 0.01

#: How close an opponent must be to take the ball off the side that has it.
#:
#: Tighter than the touch radius on purpose. With twenty-two players inside a
#: hundred metres somebody is nearly always within two metres of the ball, so
#: giving possession to whoever is merely nearest made it change hands several
#: times a second. A player in possession shields it; winning it back takes
#: getting properly to it.
_TACKLE_RADIUS_M: Final = 1.1

#: A carry push sends the ball this far ahead. Short enough that the carrier
#: stays within a touch radius for most of the roll.
_CARRY_PUSH_M: Final[tuple[float, float]] = (2.5, 5.5)

#: Seconds between the ball going out and the restart being taken.
_RESTART_PAUSE_S: Final[tuple[float, float]] = (1.5, 4.0)

#: Pace at which somebody fetches the ball and places it for the restart. The
#: ball is dead, but it is still a physical object being moved by a person.
_RESTART_WALK_MS: Final = 4.0

#: Distance from goal inside which a player will shoot given a reasonable angle.
#:
#: Wider than the penalty area, because shots are taken from its edge. The
#: appetite in each profile then decides how often; between them they produce
#: roughly the reference's two dozen shots per 90 minutes.
_SHOOTING_RANGE_M: Final = 26.0

#: Seconds a receiver keeps the ball at their feet before doing anything with it.
#:
#: Without it a pass and the pass after it happen on the same frame: the
#: receiver arrives, `_resolve_touches` calls straight into `_attacker_action`,
#: and nobody ever holds the ball. That put possession spells at a median of
#: 1.8-2.7 s against a reference 4.10, and was a large part of ~4,200 events per
#: 90 against a reference 1,700-2,000.
_CONTROL_TIME_S: Final[tuple[float, float]] = (0.9, 2.4)

# ------------------------------------------------------------------ the keeper
#
# A goalkeeper is not a fielder with a different top speed, so the shape
# calculation cannot place one. These describe the one thing they actually do
# off the ball: stand between the ball and the goal, at a depth that depends on
# how close the danger is.

#: How far off the line they stand with the ball at the far end.
_GK_LINE_STANDOFF_M: Final = 1.5

#: Ball distance at which they begin to come out. Beyond it they are on their
#: line; inside it the standoff grows toward `_GK_DEPTH_M`.
_GK_ENGAGE_M: Final = 40.0

#: How far off the line they get at the closest approach.
_GK_DEPTH_M: Final = 7.0

#: Extra depth for a side playing a high line, scaled by the profile's
#: `line_offset_m` and applied only while the ball is in the opposition half.
#: This is what makes `high_press` sweep and `counter` stay home, out of a field
#: those profiles already set rather than a new knob.
_GK_SWEEP_M: Final = 10.0

#: They cover the goalmouth, not the width of the pitch. Roughly the six-yard
#: box plus a stride.
_GK_LATERAL_LIMIT_M: Final = 9.0

#: Off-ball movement, as two harmonics per axis so the path is not a sine.
_OFF_BALL_HARMONICS: Final = 2
_OFF_BALL_AMPLITUDE_M: Final[tuple[float, float]] = (1.2, 3.4)
_OFF_BALL_FREQ_HZ: Final[tuple[float, float]] = (0.05, 0.13)

#: Time constant of each player's view of where play is, so team shape follows
#: the run of play rather than the ball. Reference median player speed is
#: 1.50 m/s; a shape that tracked a 15 m/s pass directly would move far faster.
#:
#: The keeper is the exception and reads the ball almost directly. Theirs was
#: the slowest of any role at 2.6 s, which meant their target lagged play by
#: two and a half seconds — one of three reasons they never moved.
_ROLE_FOLLOW_TAU_S: Final[dict[str, float]] = {"gk": 0.9, "def": 1.1, "mid": 1.5, "fwd": 2.1}

#: Floor on the smoothing, after the profile's press scaling and the per-player
#: draw. At the old 0.2 a `high_press` defender reached ~0.36 s and the block
#: tracked the ball almost rigidly, which is why that profile ran 17.2 km per 90
#: per player. A pressing side reacts fast; it does not snap.
_MIN_FOLLOW_TAU_S: Final = 1.0

#: Seconds a player takes to close the distance to where they want to be.
#:
#: Effectively how urgent they are. At 0.3 a player two metres from their
#: target asks for 6.7 m/s, so everyone sprinted everywhere and the speed
#: distribution came out bimodal — still or flat out, with no jogging between.
#: The reference median is 1.50 m/s.
_ARRIVAL_TIME_S: Final = 1.15

#: Urgency for a player far from the ball, and the distance at which it applies.
#:
#: One arrival time for everybody meant a player forty metres away adjusting to
#: a shift in shape sprinted there, which is why outfield distance came out at
#: 11.7-17.2 km per 90 against a real 10-11. Involvement should set urgency:
#: near the ball you work, away from it you drift into position.
_INVOLVED_RADIUS_M: Final = 25.0
_IDLE_ARRIVAL_TIME_S: Final = 2.8

#: A player within this of where they want to be simply stands there.
#:
#: Without it nobody is ever still: the steering always asks for some velocity,
#: so median player speed came out at 3.4 m/s against a reference 1.50, and
#: essentially no frame had anyone below walking pace where the reference has
#: 12%. Real players spend a lot of a match standing.
_SETTLE_RADIUS_M: Final = 1.6

#: The same, for a goalkeeper, who adjusts in short steps.
#:
#: A keeper shifting across their goal moves a metre at a time, and the outfield
#: radius erased all of it: the keeper's target almost never left the circle, so
#: they were stationary for 85-93% of frames and covered under half a kilometre
#: per 90 against a real 4-5.
_GK_SETTLE_RADIUS_M: Final = 0.6

#: 4-4-2 as fractions of half-length and half-width, attacking +x. Index 0 is
#: the goalkeeper.
FORMATION_442: Final[tuple[tuple[float, float], ...]] = (
    (-0.94, 0.00),
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

_ROLE_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("gk", 0, 1),
    ("def", 1, 5),
    ("mid", 5, 9),
    ("fwd", 9, 11),
)

N_PLAYERS: Final = len(FORMATION_442)


def magnitude(vector: np.ndarray) -> float:
    """Length of a two-dimensional vector.

    A named helper rather than ``np.hypot(*v)`` at every call site: unpacking an
    array of unknown length into a two-argument ufunc is something a type
    checker cannot resolve, and the alternative — indexing twice inline — reads
    worse than the thing it computes.
    """
    return float(np.hypot(vector[0], vector[1]))


class Phase(Enum):
    """Whether the ball is live, and why it is not."""

    LIVE = auto()
    DEAD = auto()


@dataclass(slots=True)
class _PlayerMotion:
    """Per-player movement limits and quirks, drawn once per match."""

    max_speed: np.ndarray
    max_accel: np.ndarray
    max_decel: np.ndarray
    turn_base: np.ndarray
    follow_tau: np.ndarray
    settle: np.ndarray
    off_ball_amp: np.ndarray
    off_ball_freq: np.ndarray
    off_ball_phase: np.ndarray

    def drift(self, now: float) -> np.ndarray:
        """Off-ball offset for every player at match time ``now``."""
        waves = self.off_ball_amp * np.sin(
            2.0 * math.pi * self.off_ball_freq * now + self.off_ball_phase
        )
        return np.asarray(waves.sum(axis=2))


def _draw_motion(rng: random.Random, profile: FixtureProfile, *, pressing: bool) -> _PlayerMotion:
    """Draw one team's movement constants in a fixed order."""
    roles = ["mid"] * N_PLAYERS
    for name, start, stop in _ROLE_BANDS:
        for index in range(start, min(stop, N_PLAYERS)):
            roles[index] = name

    press = profile.press_reaction_scale if pressing else 1.0
    speed = np.empty(N_PLAYERS)
    accel = np.empty(N_PLAYERS)
    turn = np.empty(N_PLAYERS)
    tau = np.empty(N_PLAYERS)
    amp = np.empty((N_PLAYERS, 2, _OFF_BALL_HARMONICS))
    freq = np.empty((N_PLAYERS, 2, _OFF_BALL_HARMONICS))
    phase = np.empty((N_PLAYERS, 2, _OFF_BALL_HARMONICS))

    settle = np.empty(N_PLAYERS)

    lo_a, hi_a = _OFF_BALL_AMPLITUDE_M
    lo_f, hi_f = _OFF_BALL_FREQ_HZ
    for i in range(N_PLAYERS):
        role = roles[i]
        speed[i] = _ROLE_SPEED_MS[role] * profile.player_speed_scale * rng.uniform(0.94, 1.06)
        accel[i] = _ROLE_ACCEL_MS2[role] * profile.accel_scale * rng.uniform(0.90, 1.10)
        turn[i] = _TURN_BASE_RAD_S * rng.uniform(0.85, 1.15)
        tau[i] = _ROLE_FOLLOW_TAU_S[role] * press * rng.uniform(0.55, 1.75)
        settle[i] = _GK_SETTLE_RADIUS_M if role == "gk" else _SETTLE_RADIUS_M
        # The keeper's shuffle. Lower than an outfielder's roaming, but no
        # longer a quarter of it: the angle line now supplies the real movement
        # and this is the adjustment on top of it.
        scale = 0.6 if role == "gk" else 1.0
        for axis in range(2):
            axis_scale = 0.65 if axis == 0 else 1.0
            for harmonic in range(_OFF_BALL_HARMONICS):
                taper = 1.0 if harmonic == 0 else 0.55
                amp[i, axis, harmonic] = rng.uniform(lo_a, hi_a) * scale * axis_scale * taper
                freq[i, axis, harmonic] = rng.uniform(lo_f, hi_f) * (1.0 + 1.7 * harmonic)
                phase[i, axis, harmonic] = rng.uniform(0.0, 2.0 * math.pi)

    np.clip(speed, 0.5, MAX_PLAYER_SPEED_MS, out=speed)
    return _PlayerMotion(
        max_speed=speed,
        max_accel=accel,
        max_decel=accel * _DECEL_RATIO,
        turn_base=turn,
        follow_tau=np.clip(tau, _MIN_FOLLOW_TAU_S, 6.0),
        settle=settle,
        off_ball_amp=amp,
        off_ball_freq=freq,
        off_ball_phase=phase,
    )


def steer(
    pos: np.ndarray,
    vel: np.ndarray,
    target: np.ndarray,
    motion: _PlayerMotion,
    dt: float,
    arrival: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one team a frame under acceleration and turning limits.

    Three limits in order: a desired velocity toward the target that eases off
    over the last stride, an acceleration cap with a larger one for slowing
    down, and a yaw-rate cap that tightens with speed.

    Args:
        pos: Current positions, one row per player.
        vel: Current velocities, one row per player.
        target: Where each player is trying to be this frame.
        motion: The team's movement limits, including its settle radii.
        dt: Timestep in seconds.
        arrival: Seconds each player is willing to take to reach their target.
            Per-player rather than the single `_ARRIVAL_TIME_S`, because urgency
            depends on involvement — see `_Sim._arrival`.

    Returns:
        The new positions and velocities.
    """
    delta = target - pos
    dist = np.hypot(delta[:, 0], delta[:, 1])
    safe = np.maximum(dist, 1e-9)

    wanted = np.minimum(motion.max_speed, dist / arrival)
    wanted = np.where(dist < motion.settle, 0.0, wanted)
    desired = delta / safe[:, None] * wanted[:, None]

    dv = desired - vel
    speeding_up = np.hypot(desired[:, 0], desired[:, 1]) >= np.hypot(vel[:, 0], vel[:, 1])
    limit = np.where(speeding_up, motion.max_accel, motion.max_decel) * dt
    dv_mag = np.hypot(dv[:, 0], dv[:, 1])
    scale = np.where(dv_mag > limit, limit / np.maximum(dv_mag, 1e-9), 1.0)
    candidate = vel + dv * scale[:, None]

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


@dataclass(slots=True)
class _Pending:
    """An event whose outcome is not yet known.

    Passes and shots are recorded the instant they are struck, because that is
    when they become observable, but where they end is decided later by whoever
    touches the ball next or by it leaving the pitch. Held here until then.
    """

    team: Team
    type: EventType
    subtype: str | None
    raw_type: str
    period: int
    start_frame: int
    start_time_s: float
    start_xy: tuple[float, float]
    from_player: str | None


@dataclass(slots=True)
class SimulatedPeriod:
    """One period of tracking and the events that produced it."""

    home_xy: np.ndarray
    away_xy: np.ndarray
    ball_xy: np.ndarray
    events: list[Event] = field(default_factory=list[Event])
    #: Ball crossings into the attacking penalty area, with the team in
    #: possession, in match seconds. Derived independently of the label builder.
    true_entries: list[tuple[float, Team]] = field(default_factory=list[tuple[float, Team]])


class _Sim:
    """The state machine that produces one period.

    Holds the ball, both teams and the possession state, and steps them one
    frame at a time. Written as a class rather than a function because the
    frame loop consults and mutates a dozen pieces of state, and threading them
    through free functions obscured which of them a given rule actually reads.
    """

    def __init__(
        self,
        rng: random.Random,
        profile: FixtureProfile,
        pitch: Pitch,
        *,
        period: int,
        frame_offset: int,
        start_s: float,
        frame_rate: float,
        attack_sign: dict[Team, float],
    ) -> None:
        """Place both teams in shape and put the ball on the centre spot."""
        self._rng = rng
        self._profile = profile
        self._pitch = pitch
        self._period = period
        self._frame_offset = frame_offset
        self._start_s = start_s
        self._dt = 1.0 / frame_rate
        self._rate = frame_rate
        self._sign = attack_sign

        self._motion = {
            Team.HOME: _draw_motion(rng, profile, pressing=False),
            Team.AWAY: _draw_motion(rng, profile, pressing=True),
        }
        self._pos = {
            team: self._shape(team, np.zeros((N_PLAYERS, 2)), attacking=False, drift=None)
            for team in (Team.HOME, Team.AWAY)
        }
        self._vel = {team: np.zeros((N_PLAYERS, 2)) for team in (Team.HOME, Team.AWAY)}
        self._view = {team: np.zeros((N_PLAYERS, 2)) for team in (Team.HOME, Team.AWAY)}
        self._alpha = {
            team: (1.0 - np.exp(-self._dt / self._motion[team].follow_tau))[:, None]
            for team in (Team.HOME, Team.AWAY)
        }

        self._ball = np.zeros(2)
        self._ball_vel = np.zeros(2)
        self._phase = Phase.LIVE
        self._dead_until = 0
        self._restart_at = np.zeros(2)
        self._restart_team = Team.HOME
        #: Frames during which only the side awarded the restart may touch the
        #: ball. Without it the nearest player of either team takes the throw.
        self._restart_grace = 0
        self._restart_grace_s = 6.0
        #: Whoever touched the ball last owns it. Possession is not scheduled.
        self._holder: tuple[Team, int] | None = None
        #: Who a pass was aimed at and where. They run onto it; whether they
        #: get there first is what decides if the pass is completed, so this
        #: is an intention, never an outcome.
        self._intended: tuple[Team, int] | None = None
        #: Frame until which whoever received the ball is settling it before
        #: doing anything with it.
        self._control_until = 0
        self._intended_at = np.zeros(2)
        self._pending: _Pending | None = None
        self._in_box_prev = False
        self._events: list[Event] = []
        self._entries: list[tuple[float, Team]] = []
        self._frame = 0

    # ------------------------------------------------------------ geometry

    def _shape(
        self, team: Team, ball_view: np.ndarray, *, attacking: bool, drift: np.ndarray | None
    ) -> np.ndarray:
        """Formation targets for one team, one ball position per player."""
        sign = self._sign[team]
        template = np.asarray(FORMATION_442, dtype=np.float64)
        xy = np.empty_like(template)
        xy[:, 0] = template[:, 0] * self._pitch.half_length
        xy[:, 1] = template[:, 1] * self._pitch.half_width * self._profile.width_scale
        xy *= sign

        progress = ball_view[:, 0] * sign
        if attacking:
            shift = 0.55 * progress
        else:
            shift = 0.45 * progress + 8.0 + self._profile.line_offset_m
        xy[1:, 0] += shift[1:] * sign

        squeeze = 0.25 if attacking else self._profile.press_squeeze
        xy[1:, 1] = xy[1:, 1] * (1.0 - squeeze) + ball_view[1:, 1] * squeeze

        if drift is not None:
            xy += drift
            xy[0] = self._keeper_target(team, ball_view[0]) + drift[0]
        else:
            xy[0] = self._keeper_target(team, ball_view[0])

        half_l, half_w = self._pitch.half_length, self._pitch.half_width
        np.clip(xy[:, 0], -half_l + 1.0, half_l - 1.0, out=xy[:, 0])
        np.clip(xy[:, 1], -half_w + 1.0, half_w - 1.0, out=xy[:, 1])
        return xy

    def _keeper_target(self, team: Team, ball_view: np.ndarray) -> np.ndarray:
        """Where ``team``'s goalkeeper wants to be.

        Its own method because a keeper is not a fielder with a different top
        speed, and the formation calculation cannot place one. They stand on the
        line between the ball and their own goal, at a depth that grows as the
        ball comes closer — narrowing the angle — and they cover the goalmouth
        rather than the width of the pitch.

        The previous version pinned x to a constant and tracked 22% of a heavily
        lagged ball position, which produced exactly zero metres of forward
        travel across a whole period.
        """
        own_goal = self._goal_centre(team.opponent)
        to_ball = ball_view - own_goal
        distance = magnitude(to_ball)
        if distance < 1e-6:
            return np.asarray(own_goal + np.array([self._sign[team] * _GK_LINE_STANDOFF_M, 0.0]))
        unit = to_ball / distance

        closeness = min(max(1.0 - distance / _GK_ENGAGE_M, 0.0), 1.0)
        depth = _GK_DEPTH_M * closeness

        # A side with play in the other half leaves ground behind it, and the
        # keeper is who covers it. How far they step up is the profile's:
        # `high_press` pushes right out, `counter` stays on its line, and a
        # patient side is somewhere between. Only while play is in the far half
        # — a sweeper keeper defends space, they do not stand there once the
        # ball comes back.
        in_far_half = float(ball_view[0]) * self._sign[team] > 0.0
        if in_far_half:
            step_up = min(max(0.4 + self._profile.line_offset_m / 14.0, 0.0), 1.0)
            depth += _GK_SWEEP_M * step_up

        target = np.asarray(own_goal + unit * (_GK_LINE_STANDOFF_M + depth))
        target[1] = float(np.clip(target[1], -_GK_LATERAL_LIMIT_M, _GK_LATERAL_LIMIT_M))
        return target

    def _in_own_area(self, team: Team, point: np.ndarray) -> bool:
        """Whether ``point`` is inside the penalty area ``team`` defends."""
        own_goal_x = self._pitch.half_length * -self._sign[team]
        inside_x = abs(float(point[0]) - own_goal_x) <= self._pitch.penalty_area_depth
        return inside_x and abs(float(point[1])) <= self._pitch.penalty_area_y_abs_max

    def _goal_centre(self, team: Team) -> np.ndarray:
        """The goal ``team`` is attacking."""
        return np.array([self._pitch.half_length * self._sign[team], 0.0])

    def _distances(self, team: Team) -> np.ndarray:
        """Every player of ``team``'s distance to the ball."""
        delta = self._pos[team] - self._ball
        return np.hypot(delta[:, 0], delta[:, 1])

    # ------------------------------------------------------------ the ball

    def _touch(
        self,
        team: Team,
        player: int,
        velocity: np.ndarray,
        kind: EventType,
        raw: str,
        subtype: str | None = None,
    ) -> None:
        """Put a foot on the ball. The only place its velocity is ever written.

        Every direction change the ball makes passes through here and leaves an
        event behind, which is what makes "the ball did not turn without a
        cause" a property of the code rather than a hope.
        """
        speed = magnitude(velocity)
        if speed > MAX_BALL_SPEED_MS:
            velocity = velocity * (MAX_BALL_SPEED_MS / speed)

        self._resolve_pending(team, player)
        self._ball_vel = velocity
        self._holder = (team, player)
        self._intended = None

        # Recorded from the ball's own position, so a pass origin is where the
        # ball actually was rather than where the passer happened to be.
        origin = (float(self._ball[0]), float(self._ball[1]))
        self._pending = _Pending(
            team=team,
            type=kind,
            subtype=subtype,
            raw_type=raw,
            period=self._period,
            start_frame=self._frame_offset + self._frame,
            start_time_s=self._start_s + self._frame * self._dt,
            start_xy=origin,
            from_player=f"{team.value}_{player + 1}",
        )

    def _resolve_pending(self, team: Team | None, player: int | None) -> None:
        """Close the open event, now that its outcome is known."""
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        self._events.append(
            Event(
                team=pending.team,
                type=pending.type,
                subtype=pending.subtype,
                period=pending.period,
                start_frame=pending.start_frame,
                end_frame=self._frame_offset + self._frame,
                start_time_s=pending.start_time_s,
                end_time_s=self._start_s + self._frame * self._dt,
                from_player=pending.from_player,
                to_player=(
                    f"{team.value}_{player + 1}"
                    if team is not None and player is not None and team is pending.team
                    else None
                ),
                start_xy=pending.start_xy,
                end_xy=(float(self._ball[0]), float(self._ball[1])),
                raw_type=pending.raw_type,
            )
        )

    def _record(self, team: Team, kind: EventType, raw: str, subtype: str | None = None) -> None:
        """Record an instantaneous event that touches nothing — out of play, restarts."""
        self._resolve_pending(None, None)
        frame = self._frame_offset + self._frame
        time_s = self._start_s + self._frame * self._dt
        here = (float(self._ball[0]), float(self._ball[1]))
        self._events.append(
            Event(
                team=team,
                type=kind,
                subtype=subtype,
                period=self._period,
                start_frame=frame,
                end_frame=frame,
                start_time_s=time_s,
                end_time_s=time_s,
                from_player=None,
                to_player=None,
                start_xy=here,
                end_xy=here,
                raw_type=raw,
            )
        )

    # ------------------------------------------------------------ decisions

    def _pass_target(self, team: Team, player: int) -> tuple[int, np.ndarray] | None:
        """Pick a team-mate to pass to, and where to put the ball.

        Prefers options that are ahead, far enough from an opponent to be
        reachable, and within the reference pass-length range. Returns the
        lead position rather than the receiver's feet, so a pass is played into
        space the way a real one is.
        """
        sign = self._sign[team]
        mates = self._pos[team]
        opponents = self._pos[team.opponent]

        best: tuple[float, int, np.ndarray] | None = None
        for mate in range(1, N_PLAYERS):
            if mate == player:
                continue
            delta = mates[mate] - self._ball
            distance = magnitude(delta)
            if not _PASS_LENGTH_M[0] <= distance <= _PASS_LENGTH_M[1]:
                continue
            # Lead the receiver, in the direction they are already going.
            lead = mates[mate] + self._vel[team][mate] * 0.45
            gap = np.hypot(opponents[:, 0] - lead[0], opponents[:, 1] - lead[1]).min()
            progress = float((lead[0] - self._ball[0]) * sign)
            score = gap + progress * self._profile.directness
            if best is None or score > best[0]:
                best = (score, mate, lead)
        if best is None:
            return None
        _, mate, lead = best
        return mate, lead

    def _launch(self, origin: np.ndarray, target: np.ndarray, flight_s: float) -> np.ndarray:
        """Velocity that carries the ball from ``origin`` to ``target``.

        Solved against the drag the ball actually experiences rather than
        assuming a constant speed, so a pass arrives roughly when intended and
        the flight-time distribution stays near the reference median of 1.2 s.
        """
        delta = target - origin
        distance = magnitude(delta)
        if distance < 1e-6:
            return np.zeros(2)
        # Integrating v(t) = v0 * exp(-k t) over the flight gives the distance
        # covered; inverting it gives the launch speed for a wanted duration.
        decay = 1.0 - math.exp(-_BALL_DRAG_PER_S * flight_s)
        speed = distance * _BALL_DRAG_PER_S / max(decay, 1e-6)
        launched: np.ndarray = delta / distance * min(speed, MAX_BALL_SPEED_MS)
        return launched

    def _attacker_action(self, team: Team, player: int) -> None:
        """Decide what a player in possession does with the ball."""
        if player == 0:
            self._distribute(team, player)
            return

        goal = self._goal_centre(team)
        to_goal = goal - self._ball
        range_to_goal = magnitude(to_goal)
        opponents = self._pos[team.opponent]
        gaps = np.hypot(opponents[:, 0] - self._ball[0], opponents[:, 1] - self._ball[1])
        pressure = float(gaps.min())

        shooting = (
            range_to_goal < _SHOOTING_RANGE_M
            and abs(self._ball[1]) < self._pitch.penalty_area_y_abs_max
            and self._rng.random() < self._profile.shot_appetite
        )
        if shooting:
            aim = goal + np.array([0.0, self._rng.uniform(-4.5, 4.5)])
            velocity = self._launch(self._ball, aim, 0.6)
            self._touch(team, player, velocity, EventType.SHOT, "SHOT", "ON TARGET")
            return

        # Under real pressure with no time, the ball goes long rather than
        # through a crowd. This is what produces clearances without a rule that
        # says "sometimes clear".
        if pressure < 2.5 and self._rng.random() < 0.45:
            self._clear(team, player)
            return

        option = self._pass_target(team, player)
        if option is not None and self._rng.random() < self._profile.pass_appetite:
            # The receiver's identity is not recorded here on purpose. Who the
            # ball reaches is decided by who gets to it, which may not be the
            # intended target — that is what makes an interception possible
            # rather than scheduled.
            mate, lead = option
            flight = self._rng.uniform(0.8, 1.9)
            self._touch(
                team,
                player,
                self._launch(self._ball, lead, flight),
                EventType.PASS,
                "PASS",
                None,
            )
            self._intended = (team, mate)
            self._intended_at = lead
            return

        self._carry(team, player)

    def _carry(self, team: Team, player: int) -> None:
        """Push the ball ahead and run after it.

        A dribble is this repeated: the ball rolls straight between pushes and
        the carrier catches up, which is both how it works and why the ball's
        path stays piecewise straight.
        """
        sign = self._sign[team]
        heading = self._vel[team][player]
        if magnitude(heading) < 0.5:
            heading = np.array([sign, 0.0])
        heading = heading / max(magnitude(heading), 1e-9)
        # Bias the push goalward so carries progress rather than wander.
        heading = heading + np.array([sign * 0.3, 0.0])
        heading = heading / max(magnitude(heading), 1e-9)
        push = self._rng.uniform(*_CARRY_PUSH_M)
        velocity = self._launch(self._ball, self._ball + heading * push, 0.7)
        self._touch(team, player, velocity, EventType.CARRY, "CARRY", None)

    def _distribute(self, team: Team, player: int) -> None:
        """A goalkeeper with the ball puts it back into play.

        Long and upfield like a clearance, because that is the shape of the
        delivery, but recorded as its own thing: the event log should not call a
        goal kick a panic hoof. Aimed at a team-mate when there is one to aim
        at, so distribution can be intercepted like any other pass.
        """
        option = self._pass_target(team, player)
        if option is not None:
            mate, lead = option
            self._touch(
                team,
                player,
                self._launch(self._ball, lead, 1.4),
                EventType.PASS,
                "DISTRIBUTION",
                "DISTRIBUTION",
            )
            self._intended = (team, mate)
            self._intended_at = lead
            return

        sign = self._sign[team]
        distance = self._rng.uniform(*_CLEARANCE_LENGTH_M)
        aim = self._ball + np.array(
            [sign * distance, self._rng.uniform(-1.0, 1.0) * distance * 0.35]
        )
        self._touch(
            team,
            player,
            self._launch(self._ball, aim, 1.6),
            EventType.PASS,
            "DISTRIBUTION",
            "DISTRIBUTION",
        )

    def _clear(self, team: Team, player: int) -> None:
        """Hoof it. Long, roughly upfield, and deliberately not to anybody."""
        sign = self._sign[team]
        distance = self._rng.uniform(*_CLEARANCE_LENGTH_M)
        aim = self._ball + np.array(
            [sign * distance, self._rng.uniform(-1.0, 1.0) * distance * 0.45]
        )
        velocity = self._launch(self._ball, aim, 1.6)
        self._touch(team, player, velocity, EventType.PASS, "CLEARANCE", "CLEARANCE")

    def _defender_action(self, team: Team, player: int) -> None:
        """A defender reaching the ball wins it, or half-wins it.

        Three outcomes, drawn once so the same situation does not resolve two
        ways: clean interception, a deflection that leaves the ball loose, or a
        challenge that fails to change possession.
        """
        incoming = magnitude(self._ball_vel)
        if player == 0 and incoming > 8.0:
            # A save. The ball comes off the keeper hard and away from goal,
            # which is a rebound: live, loose, and nobody's yet.
            away = self._ball - self._goal_centre(team.opponent)
            away = away / max(magnitude(away), 1e-9)
            spread = self._rng.uniform(-0.9, 0.9)
            direction = np.array(
                [
                    away[0] * math.cos(spread) - away[1] * math.sin(spread),
                    away[0] * math.sin(spread) + away[1] * math.cos(spread),
                ]
            )
            self._touch(
                team,
                player,
                direction * incoming * self._rng.uniform(0.35, 0.75),
                EventType.CHALLENGE,
                "SAVE",
                "SAVE-REBOUND",
            )
            return

        roll = self._rng.random()
        if roll < 0.38:
            self._record(team, EventType.RECOVERY, "RECOVERY", "INTERCEPTION")
            self._carry(team, player)
            return
        if roll < 0.44:
            # Redirected without control: a new direction, a recorded cause,
            # and the ball loose for whoever gets there next.
            speed = magnitude(self._ball_vel) * self._rng.uniform(0.3, 0.7)
            angle = self._rng.uniform(-math.pi, math.pi)
            deflected = np.array([math.cos(angle), math.sin(angle)]) * speed
            self._touch(team, player, deflected, EventType.CHALLENGE, "DEFLECTION", "DEFLECTION")
            return
        # Got a foot in but the ball runs on: possession changes, direction
        # barely does.
        self._touch(
            team,
            player,
            self._ball_vel * self._rng.uniform(0.55, 0.9),
            EventType.CHALLENGE,
            "CHALLENGE",
            None,
        )

    # ------------------------------------------------------------ boundaries

    def _check_boundary(self) -> bool:
        """Handle the ball leaving the pitch. Returns whether it did.

        A boundary is the one direction change with no player behind it, and it
        is recorded for the same reason a touch is: something caused it, and the
        record has to say what.
        """
        half_l, half_w = self._pitch.half_length, self._pitch.half_width
        x, y = float(self._ball[0]), float(self._ball[1])
        if abs(x) <= half_l and abs(y) <= half_w:
            return False

        last = self._holder[0] if self._holder else Team.HOME
        conceded = last.opponent
        # Stop it on the line rather than wherever the step happened to land.
        # The ball is out the instant it crosses, and recording it a metre
        # beyond makes every downstream bound — pitch extent, restart spot —
        # depend on the frame rate.
        self._ball = np.array(
            [float(np.clip(x, -half_l, half_l)), float(np.clip(y, -half_w, half_w))]
        )

        if abs(y) > half_w:
            # Touchline: throw-in to the side that did not put it out, taken
            # from where it crossed.
            spot = np.array([np.clip(x, -half_l + 1.0, half_l - 1.0), math.copysign(half_w, y)])
            restart_for, subtype = conceded, "THROW IN"
        elif math.copysign(1.0, x) == self._sign[last]:
            # Over the goal line the attacking team was aiming at: a corner if
            # a defender put it out, a goal kick otherwise.
            spot = np.array([math.copysign(half_l - 11.0, x), 0.0])
            restart_for, subtype = conceded, "GOAL KICK"
        else:
            spot = np.array([math.copysign(half_l - 11.0, x), 0.0])
            restart_for, subtype = conceded, "GOAL KICK"

        self._record(last, EventType.BALL_OUT, "BALL OUT", subtype)
        self._phase = Phase.DEAD
        self._ball_vel = np.zeros(2)
        self._restart_at = spot
        self._restart_team = restart_for
        self._dead_until = self._frame + int(self._rng.uniform(*_RESTART_PAUSE_S) * self._rate)
        self._holder = None
        self._control_until = 0
        return True

    def _walk_to_restart(self) -> None:
        """Carry the ball to the restart spot at walking pace.

        Not decoration. Teleporting it there put a 361 m/s step in the tracking,
        which is a discontinuity no camera or player could produce and which any
        honest continuity check has to reject. Somebody physically fetches the
        ball and places it, so the simulation does too.
        """
        delta = self._restart_at - self._ball
        distance = magnitude(delta)
        if distance < 1e-6:
            return
        step = min(distance, _RESTART_WALK_MS * self._dt)
        self._ball = self._ball + delta / distance * step

    def _restart(self) -> None:
        """Put the ball back in play with a set piece."""
        self._ball_vel = np.zeros(2)
        self._phase = Phase.LIVE
        self._record(self._restart_team, EventType.SET_PIECE, "SET PIECE", "RESTART")
        # Nobody is placed on the ball and nothing is played for them. The ball
        # sits until a player actually reaches it, exactly as in open play —
        # teleporting a taker onto the spot broke every motion bound at once
        # and recorded the resulting touch six metres from the ball.
        self._holder = None
        self._intended = None
        self._control_until = 0
        self._restart_grace = self._frame + int(self._restart_grace_s * self._rate)

    # ------------------------------------------------------------ the loop

    def _targets(self, team: Team, attacking: bool, now: float) -> np.ndarray:
        """Where every player of ``team`` is trying to be this frame."""
        motion = self._motion[team]
        view = self._view[team]
        view += self._alpha[team] * (self._ball - view)
        shape = self._shape(team, view, attacking=attacking, drift=motion.drift(now))

        # Whoever is nearest goes to the ball — chasing it down in possession,
        # closing it down out of possession. Everyone else holds shape.
        distances = self._distances(team)
        chaser = int(np.argmin(distances[1:]) + 1)
        # The keeper competes for it only inside their own area, and only when
        # it is not already someone's. That is the difference between coming for
        # a cross and running to the halfway line: gate on the area, not on who
        # happens to be nearest, or the keeper leaves their goal every time play
        # is in their half.
        loose = self._holder is None or self._holder[0] is not team
        if loose and self._in_own_area(team, self._ball) and distances[0] < distances[chaser]:
            chaser = 0
        lead = self._ball + self._ball_vel * 0.35
        shape[chaser] = lead
        if self._phase is Phase.DEAD and team is not self._restart_team:
            # The other side backs off a set piece rather than crowding it.
            shape[chaser] = self._pos[team][chaser]
        if self._intended is not None and self._intended[0] is team:
            shape[self._intended[1]] = self._intended_at
        return shape

    def _arrival(self, team: Team) -> np.ndarray:
        """How urgently each player of ``team`` closes on their target.

        Involvement sets urgency. Near the ball a player is in the game and
        moves at the base arrival time; far from it they are adjusting their
        position and amble there. A single arrival time for everyone had players
        forty metres from the ball sprinting to keep shape, which is most of why
        outfield distance came out half again as high as a real match.
        """
        distances = self._distances(team)
        far = np.clip(distances / _INVOLVED_RADIUS_M, 0.0, 1.0)
        return _ARRIVAL_TIME_S + far * (_IDLE_ARRIVAL_TIME_S - _ARRIVAL_TIME_S)

    def step(self, index: int, now: float) -> None:
        """Advance one frame."""
        self._frame = index

        if self._phase is Phase.DEAD:
            self._walk_to_restart()
            # Both conditions: the pause has to have elapsed *and* the ball has
            # to have arrived. Restarting on the timer alone would snap it the
            # last few metres, which is the discontinuity the walk exists to
            # remove.
            placed = magnitude(self._restart_at - self._ball) < 1e-6
            if index >= self._dead_until and placed:
                self._restart()
        else:
            self._ball = self._ball + self._ball_vel * self._dt
            self._ball_vel = self._ball_vel * (1.0 - _BALL_DRAG_PER_S * self._dt)
            if not self._check_boundary():
                self._resolve_touches()

        attacking_team = self._holder[0] if self._holder else None
        for team in (Team.HOME, Team.AWAY):
            targets = self._targets(team, attacking_team is team, now)
            self._pos[team], self._vel[team] = steer(
                self._pos[team],
                self._vel[team],
                targets,
                self._motion[team],
                self._dt,
                self._arrival(team),
            )

        self._track_entry(now)

    def _resolve_touches(self) -> None:
        """Give the ball to whoever has reached it, if anybody has."""
        holder_team = self._holder[0] if self._holder else None
        candidates: list[tuple[float, Team, int]] = []
        for team in (Team.HOME, Team.AWAY):
            distances = self._distances(team)
            # An opponent has to get properly to the ball; the side in
            # possession only has to be near it.
            reach = (
                TOUCH_RADIUS_M if team is holder_team or holder_team is None else _TACKLE_RADIUS_M
            )
            for player in range(N_PLAYERS):
                if distances[player] <= reach:
                    candidates.append((float(distances[player]), team, player))
        if not candidates:
            self._maybe_deflect()
            return

        candidates.sort(key=lambda item: (item[0], item[1].value, item[2]))
        _, team, player = candidates[0]
        # Still the same foot on the same ball: a carry is re-pushed only once
        # the ball has run away and been caught again, or every frame would be
        # a touch.
        same_foot = self._holder is not None and self._holder == (team, player)
        if same_foot and magnitude(self._ball_vel) > 1.2:
            return
        if self._holder is None or self._holder[0] is team:
            # Settling the ball takes a moment, and until it is settled the
            # player does nothing with it. An opponent can still come and take
            # it — that is the branch below, and it is why this guard sits here
            # rather than at the top of the method.
            if self._frame < self._control_until:
                return
            if not same_foot and self._holder is not None and magnitude(self._ball_vel) > 1.2:
                self._receive(team, player)
                return
            self._attacker_action(team, player)
        else:
            self._defender_action(team, player)

    def _receive(self, team: Team, player: int) -> None:
        """Take a pass down.

        The ball comes under control and travels with the receiver, which is
        both what a first touch does and what keeps `_touch` the only writer of
        ball velocity. Recorded as a carry because that is what it becomes: the
        ball is at this player's feet and moving with them.
        """
        self._touch(
            team,
            player,
            self._vel[team][player].copy(),
            EventType.CARRY,
            "RECEPTION",
            "RECEPTION",
        )
        self._control_until = self._frame + int(self._rng.uniform(*_CONTROL_TIME_S) * self._rate)

    def _maybe_deflect(self) -> None:
        """A defender near the ball's line may get a touch on it in passing."""
        if self._holder is None or magnitude(self._ball_vel) < 4.0:
            return
        defending = self._holder[0].opponent
        distances = self._distances(defending)
        nearest = int(np.argmin(distances))
        if distances[nearest] > _DEFLECT_RADIUS_M:
            return
        # Rare on purpose, and a rate rather than a per-frame draw. A ball
        # passing near a defender usually just passes near a defender.
        if self._rng.random() > _DEFLECT_RATE_PER_S * self._dt:
            return
        speed = magnitude(self._ball_vel) * self._rng.uniform(0.4, 0.8)
        angle = math.atan2(self._ball_vel[1], self._ball_vel[0]) + self._rng.uniform(-1.2, 1.2)
        self._touch(
            defending,
            nearest,
            np.array([math.cos(angle), math.sin(angle)]) * speed,
            EventType.CHALLENGE,
            "DEFLECTION",
            "DEFLECTION",
        )

    def _track_entry(self, now: float) -> None:
        """Record genuine penalty-area entries, independently of the labeller."""
        if self._holder is None or self._phase is Phase.DEAD:
            self._in_box_prev = False
            return
        team = self._holder[0]
        oriented = self._ball * self._sign[team]
        inside = bool(self._pitch.is_inside_penalty_area(oriented))
        if inside and not self._in_box_prev:
            self._entries.append((now, team))
        self._in_box_prev = inside

    def kick_off(self, team: Team) -> None:
        """Start the period. The first touch is a pass like any other."""
        self._attacker_action(team, 9)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Current positions, for the frame arrays."""
        return self._pos[Team.HOME].copy(), self._pos[Team.AWAY].copy(), self._ball.copy()

    def finish(self) -> tuple[list[Event], list[tuple[float, Team]]]:
        """Close any open event and hand back what the period produced."""
        self._resolve_pending(None, None)
        self._events.sort(key=lambda e: (e.period, e.start_frame))
        return self._events, self._entries


def simulate_period(
    rng: random.Random,
    profile: FixtureProfile,
    *,
    period: int,
    n_frames: int,
    frame_offset: int,
    start_s: float,
    frame_rate: float,
    attack_sign: dict[Team, float],
    pitch: Pitch = DEFAULT_PITCH,
) -> SimulatedPeriod:
    """Run one period and return its tracking and events."""
    sim = _Sim(
        rng,
        profile,
        pitch,
        period=period,
        frame_offset=frame_offset,
        start_s=start_s,
        frame_rate=frame_rate,
        attack_sign=attack_sign,
    )
    home = np.empty((n_frames, N_PLAYERS, 2))
    away = np.empty((n_frames, N_PLAYERS, 2))
    ball = np.empty((n_frames, 2))

    # The opening touch: a kick-off, played like any other pass.
    sim.kick_off(Team.HOME)

    for index in range(n_frames):
        now = start_s + (index + 1) * (1.0 / frame_rate)
        sim.step(index, now)
        home[index], away[index], ball[index] = sim.snapshot()

    events, entries = sim.finish()
    return SimulatedPeriod(
        home_xy=home, away_xy=away, ball_xy=ball, events=events, true_entries=entries
    )
