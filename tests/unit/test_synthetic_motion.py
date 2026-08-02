"""How the generated players move, and how the three archetypes differ.

Two kinds of assertion, deliberately separated. Discrete structure — seeds,
ids, periods, frame counts, event ordering, ground-truth entries — is compared
exactly, because it is integer or RNG-drawn and reproduces anywhere. Motion is
compared against *bounds and directions*, because it is floating point and the
suite has to pass on both darwin/arm64 and linux/amd64.

The guardrail numbers below are not a claim that this is real football. They are
the distance travelled from a generator whose players moved in straight lines at
a constant speed, in lockstep: 93.5% of three-frame segments collinear, median
acceleration exactly zero, team-mate directions correlated at 0.99. A generator
can satisfy every statistic here and still be tactically implausible, which is
why the module docstring also names what these numbers do *not* cover.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_insights.data.synthetic import (
    MAX_BALL_SPEED_MS,
    MAX_PLAYER_SPEED_MS,
    PROFILES,
    FixtureProfile,
    SyntheticMatch,
    generate_synthetic_match,
)
from football_insights.pitch import DEFAULT_PITCH

#: Long enough for several possession sequences, short enough to run a dozen
#: times in a unit suite.
DURATION_S = 180.0
FRAME_RATE = 25.0

#: Measured on a Metrica sample match, for reference in failure messages.
METRICA = {
    "collinear": 0.279,
    "sync": 0.475,
    "at_cap": 0.002,
    "median_accel": 1.70,
    "median_heading": 1.042,
}


def _outfield(xy: np.ndarray) -> np.ndarray:
    """Drop the goalkeeper, who is not supposed to move like the others."""
    return xy[:, 1:, :]


def _kinematics(xy: np.ndarray) -> dict[str, np.ndarray]:
    """Speed, acceleration and heading change for every outfield player."""
    dt = 1.0 / FRAME_RATE
    velocity = np.diff(_outfield(xy), axis=0) / dt
    accel = np.diff(velocity, axis=0) / dt
    speed = np.hypot(velocity[..., 0], velocity[..., 1])

    norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = velocity / np.where(norm < 1e-9, np.nan, norm)
    dot = np.sum(unit[:-1] * unit[1:], axis=-1).clip(-1.0, 1.0)
    # Only where the player is actually travelling: the heading of someone
    # standing still is noise, not a turn.
    moving = (speed[:-1] > 1.0) & (speed[1:] > 1.0)
    heading = np.degrees(np.arccos(dot))
    return {
        "speed": speed[np.isfinite(speed)],
        "accel": np.hypot(accel[..., 0], accel[..., 1]),
        "heading": heading[moving & np.isfinite(heading)],
        "unit": unit,
    }


def _teammate_sync(unit: np.ndarray) -> float:
    """Mean pairwise cosine similarity of team-mate velocity directions.

    One number for "is this a team or a rigid body". Sampled once a second
    rather than every frame; the quantity moves slowly and the full computation
    is quadratic in squad size.
    """
    scores: list[float] = []
    for index in range(0, unit.shape[0], int(FRAME_RATE)):
        vectors = unit[index]
        vectors = vectors[np.isfinite(vectors).all(axis=1)]
        if len(vectors) < 4:
            continue
        gram = vectors @ vectors.T
        scores.append(float((gram.sum() - np.trace(gram)) / (len(vectors) * (len(vectors) - 1))))
    return float(np.mean(scores))


_CACHE: dict[str, SyntheticMatch] = {}


def _all() -> dict[str, SyntheticMatch]:
    """Every profile's fixture, generated once and shared by the whole module.

    Not a pytest fixture: these are consumed by parametrised cases and by module
    helpers, and generating one per case would run the simulation forty times.
    """
    if not _CACHE:
        for profile in PROFILES:
            _CACHE[profile.key] = generate_synthetic_match(
                seed=1234, n_periods=1, period_duration_s=DURATION_S, profile=profile
            )
    return _CACHE


def matches_for(profile: FixtureProfile) -> SyntheticMatch:
    """The cached fixture for one profile."""
    return _all()[profile.key]


class TestDeterminism:
    """The seed still decides everything, profile by profile."""

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_the_same_seed_and_profile_repeat_exactly(self, profile: FixtureProfile) -> None:
        first = generate_synthetic_match(
            seed=5, n_periods=1, period_duration_s=60.0, profile=profile
        )
        second = generate_synthetic_match(
            seed=5, n_periods=1, period_duration_s=60.0, profile=profile
        )

        assert np.array_equal(first.tracking.home_xy, second.tracking.home_xy, equal_nan=True)
        assert np.array_equal(first.tracking.ball_xy, second.tracking.ball_xy, equal_nan=True)
        assert first.events == second.events
        assert first.true_entries == second.true_entries

    def test_one_seed_gives_three_different_matches(self) -> None:
        """The profile has to be part of the identity, not a label on one match."""
        produced = [
            generate_synthetic_match(seed=5, n_periods=1, period_duration_s=60.0, profile=p)
            for p in PROFILES
        ]
        for i in range(len(produced)):
            for j in range(i + 1, len(produced)):
                assert not np.array_equal(
                    produced[i].tracking.ball_xy, produced[j].tracking.ball_xy, equal_nan=True
                ), f"{PROFILES[i].key} and {PROFILES[j].key} produced the same ball trace"

    def test_no_filesystem_or_network_access_is_needed(self) -> None:
        # Nothing to assert beyond the call succeeding: the generator imports
        # only numpy and the standard library, and the published container has
        # no dataset for it to read even if it wanted one.
        match = generate_synthetic_match(seed=99, n_periods=1, period_duration_s=30.0)
        assert match.tracking.n_frames == int(30.0 * FRAME_RATE)


class TestPhysicalBounds:
    """Limits the data must never cross, whatever the profile."""

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_every_coordinate_and_timestamp_is_finite(self, profile: FixtureProfile) -> None:
        match = matches_for(profile)
        assert np.isfinite(match.tracking.time_s).all()
        assert np.isfinite(match.tracking.home_xy).all()
        assert np.isfinite(match.tracking.away_xy).all()
        assert np.isfinite(match.tracking.ball_xy).all()

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_players_stay_within_the_speed_cap(self, profile: FixtureProfile) -> None:
        speed = _kinematics(matches_for(profile).tracking.home_xy)["speed"]
        assert speed.max() <= MAX_PLAYER_SPEED_MS + 1e-6

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_nobody_teleports(self, profile: FixtureProfile) -> None:
        """A displacement cap is the bound a bug in the steering would break."""
        tracking = matches_for(profile).tracking
        step = np.diff(_outfield(tracking.home_xy), axis=0)
        assert np.hypot(step[..., 0], step[..., 1]).max() <= MAX_PLAYER_SPEED_MS / FRAME_RATE + 1e-6

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_the_ball_stays_within_its_own_cap(self, profile: FixtureProfile) -> None:
        ball = matches_for(profile).tracking.ball_xy
        speed = np.hypot(*np.diff(ball, axis=0).T) * FRAME_RATE
        assert speed.max() <= MAX_BALL_SPEED_MS + 1e-6

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_the_ball_is_not_pinned_to_its_cap(self, profile: FixtureProfile) -> None:
        """The cap must bound the ball, not describe how fast it travels.

        Before the ball carried velocity it was moved straight at its target and
        clipped, which put its 99th percentile at exactly 22.00 m/s — the limit,
        to the last digit.
        """
        ball = matches_for(profile).tracking.ball_xy
        speed = np.hypot(*np.diff(ball, axis=0).T) * FRAME_RATE
        assert np.percentile(speed, 99) < MAX_BALL_SPEED_MS * 0.99

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_the_ball_stays_on_the_pitch(self, profile: FixtureProfile) -> None:
        ball = matches_for(profile).tracking.ball_xy
        assert np.abs(ball[:, 0]).max() <= DEFAULT_PITCH.half_length
        assert np.abs(ball[:, 1]).max() <= DEFAULT_PITCH.half_width

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_player_identity_is_stable(self, profile: FixtureProfile) -> None:
        tracking = matches_for(profile).tracking
        assert [p.player_id for p in tracking.home_players] == [
            f"home_{i + 1}" for i in range(tracking.home_xy.shape[1])
        ]
        assert tracking.home_players[0].is_goalkeeper
        assert not any(p.is_goalkeeper for p in tracking.home_players[1:])


class TestMotionGuardrails:
    """The five statistics that told the old generator apart from real tracking.

    One-sided bounds against the reference, not equality with it. The comparison
    that matters is with where this started, which every failure message names.
    """

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_movement_is_not_dominated_by_straight_lines(self, profile: FixtureProfile) -> None:
        heading = _kinematics(matches_for(profile).tracking.home_xy)["heading"]
        collinear = float(np.mean(heading < 0.5))
        assert collinear <= 0.45, (
            f"{collinear:.3f} of segments are near-collinear; was 0.935 before the "
            f"steering model, Metrica sits at {METRICA['collinear']}"
        )

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_the_team_is_not_one_rigid_body(self, profile: FixtureProfile) -> None:
        sync = _teammate_sync(_kinematics(matches_for(profile).tracking.home_xy)["unit"])
        assert sync <= 0.70, (
            f"team-mate directions correlate at {sync:.3f}; was 0.993 when the whole "
            f"formation translated together, Metrica sits at {METRICA['sync']}"
        )

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_players_are_not_pinned_to_the_speed_cap(self, profile: FixtureProfile) -> None:
        speed = _kinematics(matches_for(profile).tracking.home_xy)["speed"]
        at_cap = float(np.mean(speed >= MAX_PLAYER_SPEED_MS * 0.95))
        assert at_cap <= 0.02, (
            f"{at_cap:.3f} of frames are at the cap; was 0.121, Metrica {METRICA['at_cap']}"
        )

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
    def test_players_accelerate_and_turn(self, profile: FixtureProfile) -> None:
        kin = _kinematics(matches_for(profile).tracking.home_xy)
        median_accel = float(np.median(kin["accel"]))
        median_turn = float(np.median(kin["heading"]))
        assert median_accel >= 0.8, (
            f"median acceleration {median_accel:.2f} m/s^2; was exactly 0.00 when players "
            f"moved at a constant speed, Metrica {METRICA['median_accel']}"
        )
        assert median_turn >= 0.4, (
            f"median heading change {median_turn:.3f} deg/frame; was exactly 0.000, "
            f"Metrica {METRICA['median_heading']}"
        )


class TestProfilesAreDistinct:
    """Three archetypes a viewer could tell apart, not three seeds."""

    def test_possession_length_separates_the_profiles(self) -> None:
        durations = {key: _possession_seconds(match) for key, match in _all().items()}
        assert durations["build_up"] > durations["high_press"] * 1.4, (
            f"patient build-up should hold the ball far longer: {durations}"
        )
        assert durations["counter"] < durations["build_up"]

    def test_team_width_separates_the_profiles(self) -> None:
        widths = {key: _team_width(match) for key, match in _all().items()}
        assert widths["build_up"] > widths["high_press"], (
            f"a patient side should play wider than a pressing one: {widths}"
        )

    def test_every_profile_produces_the_event_the_demo_is_about(self) -> None:
        for key, match in _all().items():
            assert len(match.true_entries) > 0, f"{key} produced no penalty-area entries"

    def test_quiet_stretches_are_bounded(self) -> None:
        """No profile may leave a viewer watching nothing for long.

        The spacing rule exists because independent per-sequence draws let
        several barren sequences fall together, and a visitor who arrives during
        that run sees a demo that appears to be doing nothing.
        """
        for key, match in _all().items():
            entries = [t for t, _ in match.true_entries]
            assert entries, f"{key} produced no entries at all"
            gaps = np.diff([0.0, *entries, DURATION_S])
            assert gaps.max() <= 90.0, (
                f"{key} goes {gaps.max():.0f}s of match time without a box entry"
            )


def _possession_seconds(match: SyntheticMatch) -> float:
    """Mean seconds between the events that bookend a possession sequence."""
    starts = [e.start_time_s for e in match.events]
    return float(np.mean(np.diff(sorted(starts)))) if len(starts) > 1 else 0.0


def _team_width(match: SyntheticMatch) -> float:
    """Mean lateral spread of the outfield players."""
    xy = _outfield(match.tracking.home_xy)
    return float(np.mean(xy[:, :, 1].max(axis=1) - xy[:, :, 1].min(axis=1)))
