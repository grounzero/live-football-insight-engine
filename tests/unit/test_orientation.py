"""Attacking-direction inference.

Includes the ground-truth regression: sample game 3 declares its playing
direction, so the inference used on games 1 and 2 — which declare nothing — can
be checked against a known answer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from football_insights.data import metrica_epts
from football_insights.data.acquire import MATCHES_BY_ID
from football_insights.data.orientation import (
    MIN_AGREEMENT,
    identify_goalkeepers,
    infer_orientation,
)
from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
from football_insights.domain import AttackDirection, Event, MatchTracking, Team
from football_insights.errors import OrientationError
from football_insights.pitch import rotate_180

RAW_ROOT = MATCHES_BY_ID["Sample_Game_3"].paths(Path("data/raw"))


@pytest.fixture(scope="module")
def synthetic() -> SyntheticMatch:
    return generate_synthetic_match(seed=13, period_duration_s=300.0)


class TestInferenceMatchesTruth:
    def test_synthetic_ground_truth(self, synthetic: SyntheticMatch) -> None:
        orientation, _, _ = infer_orientation(synthetic.tracking, synthetic.events, "synthetic")
        assert orientation.directions == synthetic.orientation.directions

    def test_every_decision_is_unanimous_on_clean_data(self, synthetic: SyntheticMatch) -> None:
        orientation, _, _ = infer_orientation(synthetic.tracking, synthetic.events, "synthetic")
        for decision in orientation.report["decisions"]:
            assert decision["agreement"] >= MIN_AGREEMENT
            assert len(decision["signals"]) >= 3

    def test_report_records_evidence_and_goalkeeper_provenance(
        self, synthetic: SyntheticMatch
    ) -> None:
        orientation, _, _ = infer_orientation(synthetic.tracking, synthetic.events, "synthetic")
        report = orientation.report
        assert report["match_id"] == "synthetic"
        assert set(report["goalkeepers"]) == {"home", "away"}
        for decision in report["decisions"]:
            for signal in decision["signals"]:
                assert signal["tier"] in (1, 2, 3, 4)
                assert signal["detail"]
                assert 0.0 <= signal["margin"] <= 1.0


class TestStructuralChecks:
    """Facts of football that hold regardless of how the signals fall."""

    def test_direction_flips_at_half_time(self, synthetic: SyntheticMatch) -> None:
        orientation, _, _ = infer_orientation(synthetic.tracking, synthetic.events, "synthetic")
        for team in (Team.HOME, Team.AWAY):
            first = orientation.direction(1, team)
            second = orientation.direction(2, team)
            assert first is not second, "teams change ends at half time"

    def test_teams_never_attack_the_same_end(self, synthetic: SyntheticMatch) -> None:
        orientation, _, _ = infer_orientation(synthetic.tracking, synthetic.events, "synthetic")
        for period in orientation.periods():
            assert orientation.direction(period, Team.HOME) is not orientation.direction(
                period, Team.AWAY
            )

    def test_failure_to_flip_is_rejected(self, synthetic: SyntheticMatch) -> None:
        """A match where the second half is not mirrored must be refused."""
        tracking = synthetic.tracking
        second = tracking.period == 2
        # Un-flip the second half so both halves look identical in direction.
        home = tracking.home_xy.copy()
        away = tracking.away_xy.copy()
        ball = tracking.ball_xy.copy()
        home[second] = rotate_180(home[second])
        away[second] = rotate_180(away[second])
        ball[second] = rotate_180(ball[second])
        broken = dataclasses.replace(tracking, home_xy=home, away_xy=away, ball_xy=ball)

        events = tuple(
            dataclasses.replace(
                e,
                start_xy=tuple(rotate_180(np.array(e.start_xy))) if e.start_xy else None,
                end_xy=tuple(rotate_180(np.array(e.end_xy))) if e.end_xy else None,
            )
            if e.period == 2
            else e
            for e in synthetic.events
        )
        with pytest.raises(OrientationError, match="change ends at half time"):
            infer_orientation(broken, events, "broken")


class TestOverrides:
    def test_override_is_applied_and_recorded(self, synthetic: SyntheticMatch) -> None:
        truth = synthetic.orientation.directions
        forced = truth[(1, Team.HOME)].flipped
        # Both teams and both periods must be inverted together. Overriding one
        # team alone would have them attacking the same goal, which the
        # structural check rejects — as it should.
        overrides = {
            f"synthetic:{period}:{team.value}": truth[(period, team)].flipped.value
            for period in (1, 2)
            for team in (Team.HOME, Team.AWAY)
        }
        reasons = dict.fromkeys(overrides, "test fixture")
        orientation, _, _ = infer_orientation(
            synthetic.tracking,
            synthetic.events,
            "synthetic",
            overrides=overrides,
            override_reasons=reasons,
        )
        assert orientation.direction(1, Team.HOME) is forced
        decision = next(
            d for d in orientation.report["decisions"] if d["period"] == 1 and d["team"] == "home"
        )
        assert decision["source"] == "override"
        assert decision["override_reason"] == "test fixture"

    def test_override_without_a_reason_is_rejected_by_config(self) -> None:
        from football_insights.config import Settings

        with pytest.raises(ValueError, match="written justification"):
            Settings(direction_overrides={"m:1:home": "+x"})


class TestGoalkeeperIdentification:
    def test_declared_keeper_is_trusted(self, synthetic: SyntheticMatch) -> None:
        players, index = identify_goalkeepers(
            synthetic.tracking, synthetic.tracking.home_players, Team.HOME
        )
        assert index == 0
        assert players[0].goalkeeper_source == "synthetic"

    def test_keeper_is_inferred_when_not_declared(self, synthetic: SyntheticMatch) -> None:
        stripped = tuple(
            dataclasses.replace(p, is_goalkeeper=False, goalkeeper_source="unknown")
            for p in synthetic.tracking.home_players
        )
        players, index = identify_goalkeepers(synthetic.tracking, stripped, Team.HOME)
        assert index == 0, "the synthetic keeper occupies column 0"
        assert players[0].is_goalkeeper is True
        assert players[0].goalkeeper_source == "inferred"


@pytest.mark.requires_data
class TestGameThreeGroundTruth:
    """Sample game 3 declares its direction; games 1 and 2 do not.

    Withholding the declaration and checking the inference against it is the
    only direct evidence available that the inference applied to the other two
    matches is correct.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def game_three() -> tuple[MatchTracking, tuple[Event, ...], metrica_epts.EptsMetadata]:
        paths = RAW_ROOT
        if not paths["metadata"].is_file():
            pytest.skip("Metrica sample data not downloaded; run `make data`")
        return metrica_epts.read_match(paths["tracking"], paths["metadata"], paths["events"])

    def test_metadata_declares_direction(
        self, game_three: tuple[MatchTracking, tuple[Event, ...], metrica_epts.EptsMetadata]
    ) -> None:
        _, _, metadata = game_three
        declared = metrica_epts.declared_directions(metadata)
        assert declared, "sample game 3 must declare attack_direction_first_half"
        assert set(declared) == {
            (1, Team.HOME),
            (1, Team.AWAY),
            (2, Team.HOME),
            (2, Team.AWAY),
        }

    def test_inference_reproduces_the_declared_direction(
        self, game_three: tuple[MatchTracking, tuple[Event, ...], metrica_epts.EptsMetadata]
    ) -> None:
        tracking, events, metadata = game_three
        declared = metrica_epts.declared_directions(metadata)
        inferred, _, _ = infer_orientation(tracking, events, "Sample_Game_3", declared=None)
        assert inferred.directions == declared

    def test_declared_keepers_are_marked(
        self, game_three: tuple[MatchTracking, tuple[Event, ...], metrica_epts.EptsMetadata]
    ) -> None:
        tracking, _, _ = game_three
        for players in (tracking.home_players, tracking.away_players):
            keepers = [p for p in players if p.is_goalkeeper]
            assert len(keepers) == 1
            assert keepers[0].goalkeeper_source == "declared"

    def test_substitutes_get_stable_columns(
        self, game_three: tuple[MatchTracking, tuple[Event, ...], metrica_epts.EptsMetadata]
    ) -> None:
        tracking, _, _ = game_three
        # Squads exceed eleven, so some columns must be entirely absent early on.
        assert len(tracking.home_players) > 11
        first = tracking.home_xy[0]
        assert np.isfinite(first).all(axis=1).sum() == 11


class TestCoordinateConventions:
    def test_rotation_preserves_handedness(self) -> None:
        left_wing = np.array([10.0, 20.0])
        rotated = rotate_180(left_wing)
        # A 180 degree rotation negates both axes; a mirror would negate only x
        # and silently swap the wings.
        assert rotated[0] == -10.0
        assert rotated[1] == -20.0

    def test_direction_sign_round_trips(self) -> None:
        assert AttackDirection.POSITIVE_X.sign == 1.0
        assert AttackDirection.NEGATIVE_X.sign == -1.0
        assert AttackDirection.from_sign(-3.2) is AttackDirection.NEGATIVE_X
        assert AttackDirection.POSITIVE_X.flipped is AttackDirection.NEGATIVE_X
