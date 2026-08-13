"""Invariants the simulation must not be able to violate.

These are not statistics that should look about right; they are properties that
are either true of every frame or the simulation is wrong. The central one is
that **the ball cannot change direction without a recorded cause** — a touch, a
deflection or the boundary. A generator that curves the ball toward a waypoint
produces plausible-looking tracking in which nothing that happens has a reason,
and no amount of distributional realism fixes that.

Motion is compared against bounds rather than exact values, because it is
floating point and the suite runs on both darwin/arm64 and linux/amd64.
Discrete structure — event types, teams, frame indices, possession order — is
compared exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_insights.data.profiles import PROFILES
from football_insights.data.simulation import (
    MAX_BALL_SPEED_MS,
    MAX_PLAYER_SPEED_MS,
    TOUCH_RADIUS_M,
)
from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
from football_insights.domain import EventType

FRAME_RATE = 25.0
DURATION_S = 240.0

#: Heading change per frame above which the ball has clearly been redirected
#: rather than merely rolling. Well above the numerical noise of a straight
#: line and well below the smallest deliberate touch.
TURN_DEGREES = 5.0

#: Frames either side of an event in which its effect may appear. The ball is
#: stepped and the event recorded in the same frame, but the heading is a
#: difference between two frames, so the effect straddles the boundary.
EVENT_WINDOW = 2


@pytest.fixture(scope="module")
def match() -> SyntheticMatch:
    """One fixture, generated once, long enough to contain every event type."""
    return generate_synthetic_match(
        seed=17, n_periods=1, period_duration_s=DURATION_S, frame_rate=FRAME_RATE
    )


def _ball_kinematics(match: SyntheticMatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame ball step, speed and heading change in degrees."""
    ball = match.tracking.ball_xy
    step = np.hypot(*np.diff(ball, axis=0).T)
    velocity = np.diff(ball, axis=0) * FRAME_RATE
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    swing = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
    return step, speed, np.degrees(np.abs(swing))


def _event_frames(match: SyntheticMatch) -> set[int]:
    """Every frame at or near which something happened to the ball."""
    frames: set[int] = set()
    for event in match.events:
        for frame in (event.start_frame, event.end_frame):
            frames.update(range(frame - EVENT_WINDOW, frame + EVENT_WINDOW + 1))
    return frames


class TestBallContinuity:
    """The ball is a physical object with a continuous path."""

    def test_the_ball_never_teleports(self, match: SyntheticMatch) -> None:
        """No step may exceed what the speed cap allows in one frame.

        This is the check that caught the restart: placing the ball for a throw
        put a 361 m/s step in the tracking, because it was moved instantly
        rather than carried.
        """
        step, _, _ = _ball_kinematics(match)
        limit = MAX_BALL_SPEED_MS / FRAME_RATE
        assert step.max() <= limit + 1e-9, (
            f"largest ball step {step.max():.3f} m exceeds the {limit:.3f} m a "
            f"frame at {MAX_BALL_SPEED_MS} m/s allows"
        )

    def test_every_coordinate_is_finite(self, match: SyntheticMatch) -> None:
        assert np.isfinite(match.tracking.ball_xy).all()

    def test_the_ball_stays_on_the_pitch(self, match: SyntheticMatch) -> None:
        from football_insights.pitch import DEFAULT_PITCH

        ball = match.tracking.ball_xy
        assert np.abs(ball[:, 0]).max() <= DEFAULT_PITCH.half_length + 1e-6
        assert np.abs(ball[:, 1]).max() <= DEFAULT_PITCH.half_width + 1e-6


class TestBallDirectionHasACause:
    """The invariant the whole design exists to make structural."""

    def test_no_substantial_turn_without_a_recorded_event(self, match: SyntheticMatch) -> None:
        """A ball rolling free travels in a straight line.

        Anything else needs a cause, and the cause has to be in the event
        stream. The previous generator interpolated the ball along waypoints,
        so it curved continuously with nothing recorded anywhere — tracking
        that looks like football and describes nothing that happened.
        """
        _, speed, turn = _ball_kinematics(match)
        moving = (speed[:-1] > 0.5) & (speed[1:] > 0.5)
        marks = _event_frames(match)

        unexplained = [
            (index, float(turn[index]))
            for index in range(len(turn))
            if moving[index] and turn[index] > TURN_DEGREES and (index + 2) not in marks
        ]
        assert not unexplained, (
            f"{len(unexplained)} direction changes above {TURN_DEGREES} deg/frame have no "
            f"touch, deflection or boundary behind them; worst: "
            f"{sorted(unexplained, key=lambda pair: -pair[1])[:3]}"
        )

    def test_the_ball_never_speeds_up_on_its_own(self, match: SyntheticMatch) -> None:
        """Between touches it only loses energy to drag."""
        _, speed, _ = _ball_kinematics(match)
        marks = _event_frames(match)

        gained = [
            index
            for index in range(len(speed) - 1)
            if speed[index + 1] > speed[index] * 1.02 + 0.05 and (index + 2) not in marks
        ]
        assert not gained, f"the ball accelerated on {len(gained)} frames with nothing touching it"

    def test_direction_changes_do_happen(self, match: SyntheticMatch) -> None:
        """Guard against passing the two checks above by never moving.

        A ball that sits still has no unexplained turns either.
        """
        _, speed, turn = _ball_kinematics(match)
        moving = (speed[:-1] > 0.5) & (speed[1:] > 0.5)
        assert int((moving & (turn > TURN_DEGREES)).sum()) > 20


class TestPlayerMotionLimits:
    """Players are bound by acceleration and turning, not only by speed."""

    @staticmethod
    def _velocity(match: SyntheticMatch) -> np.ndarray:
        outfield = match.tracking.home_xy[:, 1:, :]
        return np.diff(outfield, axis=0) * FRAME_RATE

    def test_speed_stays_within_the_cap(self, match: SyntheticMatch) -> None:
        velocity = self._velocity(match)
        speed = np.hypot(velocity[..., 0], velocity[..., 1])
        assert speed.max() <= MAX_PLAYER_SPEED_MS + 1e-6

    def test_acceleration_stays_within_a_human_bound(self, match: SyntheticMatch) -> None:
        """Nobody accelerates harder than a sprinter starting.

        The bound is generous — the fastest role is drawn at about 5 m/s^2 and
        decelerates at 1.35 times that — because the point is to catch a
        steering bug that lets a player change velocity arbitrarily, not to
        pin the draw.
        """
        velocity = self._velocity(match)
        accel = np.hypot(*(np.diff(velocity, axis=0) * FRAME_RATE).transpose(2, 0, 1))
        assert accel.max() <= 12.0, f"peak acceleration {accel.max():.1f} m/s^2"

    def test_turning_rate_stays_within_a_human_bound(self, match: SyntheticMatch) -> None:
        """A player at speed arcs; only a stationary one may pivot freely."""
        velocity = self._velocity(match)
        speed = np.hypot(velocity[..., 0], velocity[..., 1])
        heading = np.arctan2(velocity[..., 1], velocity[..., 0])
        swing = np.abs(
            np.arctan2(np.sin(np.diff(heading, axis=0)), np.cos(np.diff(heading, axis=0)))
        )
        rate = np.degrees(swing) * FRAME_RATE

        # Only where the player is genuinely travelling: the heading of someone
        # walking is noise, and the model deliberately lets them turn on the spot.
        moving = (speed[:-1] > 2.0) & (speed[1:] > 2.0)
        assert rate[moving].max() <= 420.0, (
            f"peak turn rate while moving {rate[moving].max():.0f} deg/s"
        )


class TestPossession:
    """Who has the ball, and how it changes hands."""

    def test_a_touch_is_recorded_within_reach_of_the_ball(self, match: SyntheticMatch) -> None:
        """Possession is proximity, not a schedule.

        Every event naming a player must place them within a touch of where the
        ball actually was. The previous generator scheduled possession, so a
        team could be "in possession" with nobody near the ball.
        """
        index = {int(frame): i for i, frame in enumerate(match.tracking.frame)}
        offenders: list[tuple[str, float]] = []
        for event in match.events:
            if event.from_player is None or event.start_xy is None:
                continue
            row = index.get(event.start_frame)
            if row is None:
                continue
            ball = match.tracking.ball_xy[row]
            team_xy = (
                match.tracking.home_xy
                if event.from_player.startswith("home")
                else match.tracking.away_xy
            )
            player = int(event.from_player.rsplit("_", 1)[1]) - 1
            distance = float(np.hypot(*(team_xy[row, player] - ball)))
            if distance > TOUCH_RADIUS_M * 2.0:
                offenders.append((event.raw_type or event.type.value, distance))
        assert not offenders, f"touches recorded away from the ball: {offenders[:5]}"

    def test_a_pass_starts_where_the_ball_was(self, match: SyntheticMatch) -> None:
        """A pass origin is the ball's position, not the passer's.

        Recorded from the ball itself, so event geometry and tracking cannot
        disagree — which is exactly what event/frame alignment downstream
        assumes.
        """
        index = {int(frame): i for i, frame in enumerate(match.tracking.frame)}
        for event in match.events:
            if event.type is not EventType.PASS or event.start_xy is None:
                continue
            row = index.get(event.start_frame)
            if row is None:
                continue
            ball = match.tracking.ball_xy[row]
            assert float(np.hypot(ball[0] - event.start_xy[0], ball[1] - event.start_xy[1])) < 1.0

    def test_interceptions_take_the_ball_off_the_other_side(self, match: SyntheticMatch) -> None:
        """A recovery is a transition: it must follow the opposition touching it.

        Compared against the previous touch of any kind, not the previous
        *on-ball* event. A pass deflected off a defender and won back by the
        side that played it is ordinary football, and looks like two events by
        the same team if the deflection between them is skipped.
        """
        touches = [e for e in match.events if e.from_player is not None]
        recoveries = [e for e in match.events if e.type is EventType.RECOVERY]
        assert recoveries, "no interceptions occurred at all"

        # A recovery is recorded without a player, so pair each one with the
        # last touch before it.
        for recovery in recoveries:
            before = [t for t in touches if t.start_frame < recovery.start_frame]
            if not before:
                continue
            assert before[-1].team is not recovery.team, (
                f"a recovery at frame {recovery.start_frame} followed its own side's touch"
            )

    def test_the_ball_goes_out_and_comes_back(self, match: SyntheticMatch) -> None:
        """Out-of-play is modelled, and every stoppage is restarted."""
        outs = [e for e in match.events if e.type is EventType.BALL_OUT]
        restarts = [e for e in match.events if e.type is EventType.SET_PIECE]
        assert outs, "the ball never left the pitch"
        assert abs(len(outs) - len(restarts)) <= 1, (
            f"{len(outs)} stoppages against {len(restarts)} restarts"
        )


class TestEventVocabulary:
    """The simulation produces the football it claims to."""

    def test_every_modelled_action_occurs(self, match: SyntheticMatch) -> None:
        raw = {event.raw_type for event in match.events}
        for action in ("PASS", "CARRY", "RECOVERY", "CLEARANCE", "DEFLECTION", "BALL OUT"):
            assert action in raw, f"no {action} in a four-minute fixture"

    def test_shots_are_taken(self) -> None:
        """Shots are rare, so this asks for enough football to contain one.

        A shot needs a touch inside shooting range with a reasonable angle, and
        the reference rate is about two dozen per ninety minutes — a handful in
        four. Asserting one appears in a single short fixture is asserting a
        coin lands heads, and it failed exactly that way once the players
        stopped sprinting everywhere. Three profiles at fifteen minutes each is
        long enough for the claim to mean something.
        """
        taken = 0
        for profile in PROFILES:
            longer = generate_synthetic_match(
                seed=17, n_periods=1, period_duration_s=900.0, profile=profile
            )
            taken += sum(1 for e in longer.events if e.type is EventType.SHOT)
        assert taken > 0, "45 minutes of football across three archetypes produced no shot"

    def test_events_are_ordered_and_within_the_period(self, match: SyntheticMatch) -> None:
        frames = [e.start_frame for e in match.events]
        assert frames == sorted(frames)
        for event in match.events:
            assert event.end_frame >= event.start_frame


class TestDeterminism:
    """The seed decides everything."""

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_the_same_seed_repeats_exactly(self, profile: object) -> None:
        kwargs = {"seed": 5, "n_periods": 1, "period_duration_s": 60.0, "profile": profile}
        first = generate_synthetic_match(**kwargs)  # type: ignore[arg-type]
        second = generate_synthetic_match(**kwargs)  # type: ignore[arg-type]

        assert np.array_equal(first.tracking.ball_xy, second.tracking.ball_xy, equal_nan=True)
        assert np.array_equal(first.tracking.home_xy, second.tracking.home_xy, equal_nan=True)
        assert first.events == second.events
        assert first.true_entries == second.true_entries

    def test_different_seeds_diverge(self) -> None:
        first = generate_synthetic_match(seed=1, n_periods=1, period_duration_s=60.0)
        second = generate_synthetic_match(seed=2, n_periods=1, period_duration_s=60.0)
        assert not np.array_equal(first.tracking.ball_xy, second.tracking.ball_xy, equal_nan=True)

    def test_profiles_produce_different_matches_on_one_seed(self) -> None:
        produced = [
            generate_synthetic_match(seed=5, n_periods=1, period_duration_s=60.0, profile=p)
            for p in PROFILES
        ]
        for i in range(len(produced)):
            for j in range(i + 1, len(produced)):
                assert not np.array_equal(
                    produced[i].tracking.ball_xy, produced[j].tracking.ball_xy, equal_nan=True
                )

    def test_generation_needs_no_dataset_or_network(self) -> None:
        # The simulation imports numpy and the standard library only; the
        # published container has no dataset for it to read even if it wanted.
        match = generate_synthetic_match(seed=99, n_periods=1, period_duration_s=30.0)
        assert match.tracking.n_frames == int(30.0 * FRAME_RATE)
