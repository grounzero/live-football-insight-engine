"""Parsing boundaries where external `Any` becomes internal types.

These cover the two places the quality-hardening pass rewrote to remove
untyped traversal: YAML configuration loading and the EPTS JSON event stream.
Both convert data this project does not control, so the conversions are worth
pinning down rather than trusting to the type checker alone — a checker can only
see the annotations, not whether the runtime still agrees with them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from football_insights.config import Settings
from football_insights.data import metrica_epts
from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
from football_insights.domain import AttackDirection, EventType, Team
from football_insights.errors import DataValidationError


class TestYamlConfigBoundary:
    """`Settings.load` parses YAML into validated settings."""

    def test_empty_document_uses_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert Settings.load(path).window.horizon_s == Settings().window.horizon_s

    def test_values_override_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("window:\n  horizon_s: 7.0\n")
        assert Settings.load(path).window.horizon_s == 7.0

    def test_non_mapping_document_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- one\n- two\n")
        with pytest.raises(TypeError, match="must contain a mapping"):
            Settings.load(path)

    def test_non_string_keys_are_rejected(self, tmp_path: Path) -> None:
        """YAML permits ``1: x``; ``Settings(**data)`` would not survive it."""
        path = tmp_path / "int_keys.yaml"
        path.write_text("1: one\n")
        with pytest.raises(TypeError, match="keys must be strings"):
            Settings.load(path)

    def test_missing_file_still_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Settings.load(tmp_path / "absent.yaml")


def _metadata() -> metrica_epts.EptsMetadata:
    """Minimal metadata mapping one team id to each side."""
    return metrica_epts.EptsMetadata(
        frame_rate=25.0,
        team_sides={"FIFATMA": Team.HOME, "FIFATMB": Team.AWAY},
        squad=(),
        layouts=(),
        declared_first_half={Team.HOME: AttackDirection.POSITIVE_X},
        periods={1: (1, 100)},
    )


def _events_file(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"data": records}))
    return path


class TestEptsEventBoundary:
    """`read_events` converts untyped JSON into `Event` objects."""

    def test_a_complete_record_round_trips(self, tmp_path: Path) -> None:
        path = _events_file(
            tmp_path,
            [
                {
                    "team": {"id": "FIFATMA"},
                    "type": {"name": "PASS"},
                    "subtypes": {"name": "CROSS"},
                    "period": 1,
                    "start": {"frame": 10, "time": 0.4, "x": 0.5, "y": 0.5},
                    "end": {"frame": 20, "time": 0.8, "x": 0.6, "y": 0.4},
                    "from": {"id": "Player1"},
                    "to": {"id": "Player2"},
                }
            ],
        )
        (event,) = metrica_epts.read_events(path, _metadata())
        assert event.team is Team.HOME
        assert event.type is EventType.PASS
        assert event.subtype == "CROSS"
        assert (event.start_frame, event.end_frame) == (10, 20)
        assert event.start_time_s == 0.4
        assert event.end_time_s == 0.8
        assert event.from_player == "Player1"
        assert event.to_player == "Player2"
        assert event.start_xy is not None

    def test_events_for_unknown_teams_are_dropped(self, tmp_path: Path) -> None:
        path = _events_file(tmp_path, [{"team": {"id": "OTHER"}, "type": {"name": "PASS"}}])
        assert metrica_epts.read_events(path, _metadata()) == ()

    def test_absent_optional_fields_fall_back(self, tmp_path: Path) -> None:
        """A record with only a team and type must still produce an event."""
        path = _events_file(tmp_path, [{"team": {"id": "FIFATMA"}, "type": {"name": "PASS"}}])
        (event,) = metrica_epts.read_events(path, _metadata())
        assert event.period == 1
        assert event.start_frame == 0
        assert event.end_frame == 0
        assert event.start_time_s == 0.0
        assert event.from_player is None
        assert event.start_xy is None

    def test_end_time_falls_back_to_start_time(self, tmp_path: Path) -> None:
        path = _events_file(
            tmp_path,
            [
                {
                    "team": {"id": "FIFATMA"},
                    "type": {"name": "PASS"},
                    "start": {"frame": 5, "time": 1.5},
                    "end": {},
                }
            ],
        )
        (event,) = metrica_epts.read_events(path, _metadata())
        assert event.end_time_s == 1.5
        assert event.end_frame == 5

    def test_end_frame_never_precedes_start_frame(self, tmp_path: Path) -> None:
        path = _events_file(
            tmp_path,
            [
                {
                    "team": {"id": "FIFATMA"},
                    "type": {"name": "PASS"},
                    "start": {"frame": 40},
                    "end": {"frame": 10},
                }
            ],
        )
        (event,) = metrica_epts.read_events(path, _metadata())
        assert event.end_frame == 40

    def test_zero_is_a_real_coordinate(self, tmp_path: Path) -> None:
        """The centre spot must not be read as a missing position."""
        path = _events_file(
            tmp_path,
            [
                {
                    "team": {"id": "FIFATMA"},
                    "type": {"name": "PASS"},
                    "start": {"x": 0.5, "y": 0.5},
                    "end": {"x": 0, "y": 0},
                }
            ],
        )
        (event,) = metrica_epts.read_events(path, _metadata())
        assert event.end_xy is not None

    def test_subtypes_accepts_a_list(self, tmp_path: Path) -> None:
        path = _events_file(
            tmp_path,
            [
                {
                    "team": {"id": "FIFATMA"},
                    "type": {"name": "PASS"},
                    "subtypes": [{"name": "HEAD"}, {"name": "CLEARANCE"}],
                }
            ],
        )
        (event,) = metrica_epts.read_events(path, _metadata())
        assert event.subtype == "HEAD-CLEARANCE"

    def test_null_subtypes_is_none(self, tmp_path: Path) -> None:
        path = _events_file(
            tmp_path,
            [{"team": {"id": "FIFATMA"}, "type": {"name": "PASS"}, "subtypes": None}],
        )
        (event,) = metrica_epts.read_events(path, _metadata())
        assert event.subtype is None

    def test_unknown_event_type_maps_to_other(self, tmp_path: Path) -> None:
        path = _events_file(
            tmp_path, [{"team": {"id": "FIFATMA"}, "type": {"name": "SOMETHING NEW"}}]
        )
        (event,) = metrica_epts.read_events(path, _metadata())
        assert event.type is EventType.OTHER
        assert event.raw_type == "SOMETHING NEW"

    def test_missing_data_list_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "events.json"
        path.write_text(json.dumps({"nope": []}))
        with pytest.raises(DataValidationError, match="'data' list"):
            metrica_epts.read_events(path, _metadata())

    def test_a_non_numeric_frame_is_rejected(self, tmp_path: Path) -> None:
        """A structurally wrong type is a data fault, not something to coerce."""
        path = _events_file(
            tmp_path,
            [
                {
                    "team": {"id": "FIFATMA"},
                    "type": {"name": "PASS"},
                    "start": {"frame": ["not", "a", "frame"]},
                }
            ],
        )
        with pytest.raises(DataValidationError, match="expected a number"):
            metrica_epts.read_events(path, _metadata())


class TestRouteRegistration:
    """The refactor to module-level routers must not change the API surface.

    `create_app` used to declare every handler as a closure inside one
    registration function. The paths and methods below are the public contract
    documented in the module docstring and consumed by the React demo, so they
    are asserted directly rather than inferred from a handful of endpoint tests.
    """

    def test_documented_routes_are_registered(self) -> None:
        from football_insights.serving.app import create_app

        app = create_app(Settings())
        # Read the OpenAPI schema rather than walking `app.routes`: it is the
        # published contract, and it does not depend on how a given FastAPI
        # version represents an included router internally.
        schema = app.openapi()
        registered = {
            (path, method.upper())
            for path, operations in schema["paths"].items()
            for method in operations
        }
        expected = {
            ("/health", "GET"),
            ("/ready", "GET"),
            ("/capabilities", "GET"),
            ("/metrics", "GET"),
            ("/model", "GET"),
            ("/predict", "POST"),
            ("/replay/status", "GET"),
            ("/replay/matches", "GET"),
            ("/replay/control", "POST"),
            ("/replay/match", "POST"),
            ("/insights", "GET"),
            ("/insights/stream", "GET"),
        }
        assert expected <= registered

    def test_the_pipeline_routes_are_not_part_of_the_default_surface(self) -> None:
        """They exist only where a deployment has explicitly enabled them."""
        from football_insights.serving.app import create_app

        schema = create_app(Settings()).openapi()
        assert not any(path.startswith("/jobs") for path in schema["paths"])


class TestSyntheticDeterminism:
    """The generator's seed-to-output mapping is a fixture contract.

    Every test in this suite builds its data from a seed rather than a stored
    blob, so a change in the generator silently changes what every other test is
    asserting against. `generate_synthetic_match` was split into named stages
    during the code-health pass; these lock the output down so a future
    refactor cannot quietly reorder an RNG draw.
    """

    @staticmethod
    def _fingerprint(match: SyntheticMatch) -> str:
        digest = hashlib.sha256()
        tracking = match.tracking
        for array in (
            tracking.period,
            tracking.frame,
            tracking.time_s,
            tracking.home_xy,
            tracking.away_xy,
            tracking.ball_xy,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(
            repr(
                [
                    (
                        event.team,
                        event.type,
                        event.subtype,
                        event.period,
                        event.start_frame,
                        event.end_frame,
                        event.start_time_s,
                        event.end_time_s,
                        event.start_xy,
                        event.end_xy,
                    )
                    for event in match.events
                ]
            ).encode()
        )
        digest.update(repr(match.true_entries).encode())
        return digest.hexdigest()

    #: Digests recorded from the generator as it behaved before
    #: `generate_synthetic_match` was split into stages, and re-verified
    #: byte-identical afterwards.
    #:
    #: Comparing two calls in one process only proves the generator is
    #: repeatable — a refactor that changed the output *consistently* would move
    #: both calls together and pass. These pinned values are what actually
    #: catches a reordered RNG draw.
    #:
    #: If a change to the simulation is intended, update these deliberately and
    #: say so in the pull request: every fixture in the suite shifts with them.
    GOLDEN: ClassVar[dict[int, str]] = {
        3: "64c012813ff703a4a8bce6a38c74cbd7c19c7bcd54d2058ccae69fb528800404",
        7: "684407eac878817dacb1251df4cfa36806cc0a139952748b1911834ebe6441b1",
        17: "b8c2a65a0717e07be2e5337904073bcbf50fbf17b6d5396f218e81fb7e39ce6f",
        23: "f754d15a0475b2ff3d102918ba0b70b5f211bbaf00045dfd5985bc9aaf09750a",
    }

    @pytest.mark.parametrize("seed", [3, 7, 17, 23])
    def test_the_same_seed_gives_identical_output(self, seed: int) -> None:
        first = generate_synthetic_match(seed=seed, period_duration_s=45.0)
        second = generate_synthetic_match(seed=seed, period_duration_s=45.0)
        assert self._fingerprint(first) == self._fingerprint(second)

    @pytest.mark.parametrize("seed", [3, 7, 17, 23])
    def test_output_matches_the_recorded_fingerprint(self, seed: int) -> None:
        """The generator still produces exactly what the fixtures were built on."""
        match = generate_synthetic_match(seed=seed, period_duration_s=45.0)
        assert self._fingerprint(match) == self.GOLDEN[seed]

    def test_different_seeds_diverge(self) -> None:
        a = generate_synthetic_match(seed=1, period_duration_s=45.0)
        b = generate_synthetic_match(seed=2, period_duration_s=45.0)
        assert self._fingerprint(a) != self._fingerprint(b)

    def test_periods_are_contiguous_in_frames_and_time(self) -> None:
        """Frame numbers and timestamps run continuously across the half-time break."""
        match = generate_synthetic_match(seed=7, n_periods=2, period_duration_s=45.0)
        tracking = match.tracking
        assert np.all(np.diff(tracking.frame) == 1)
        assert np.all(np.diff(tracking.time_s) > 0)
        assert set(np.unique(tracking.period)) == {1, 2}

    def test_missing_frame_rate_blanks_the_ball(self) -> None:
        clean = generate_synthetic_match(seed=9, period_duration_s=45.0)
        holed = generate_synthetic_match(seed=9, period_duration_s=45.0, missing_frame_rate=0.1)
        missing_clean = int(np.isnan(clean.tracking.ball_xy[:, 0]).sum())
        missing_holed = int(np.isnan(holed.tracking.ball_xy[:, 0]).sum())
        assert missing_holed > missing_clean
