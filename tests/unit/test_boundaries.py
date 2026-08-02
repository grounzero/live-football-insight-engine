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
from tests.support import approx


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
        """Hash everything the generator produced, positions included.

        Exact, and therefore only meaningful *within one process on one machine*
        — see :meth:`_stable_fingerprint` for why.
        """
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
        digest.update(TestSyntheticDeterminism._event_repr(match))
        digest.update(repr(match.true_entries).encode())
        return digest.hexdigest()

    @staticmethod
    def _event_repr(match: SyntheticMatch) -> bytes:
        return repr(
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

    @staticmethod
    def _stable_fingerprint(match: SyntheticMatch) -> str:
        """Hash only the parts that are bit-identical on every platform.

        The simulated position arrays are deliberately excluded. They are the
        output of chained floating-point arithmetic, and the last bit of that
        arithmetic is not portable: identical code on identical numpy produces
        results differing by an ULP or so between arm64 and x86-64, which is
        enough to move a SHA-256. Pinning them by hash made this test assert the
        developer's CPU architecture, and it failed in CI for that reason alone.

        What remains — the clock, the frame index, every event and the
        ground-truth entries — is integer, RNG-drawn or derived from a frame
        number, and is exactly reproducible anywhere. It is also where a
        reordered draw shows up: events carry the waypoint coordinates each
        sequence was built from, so a change in draw order moves this digest.

        The positions are still covered, by
        :meth:`test_simulated_positions_match_recorded_statistics`, which
        compares them numerically instead of bitwise.
        """
        digest = hashlib.sha256()
        tracking = match.tracking
        for array in (tracking.period, tracking.frame, tracking.time_s):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(TestSyntheticDeterminism._event_repr(match))
        digest.update(repr(match.true_entries).encode())
        return digest.hexdigest()

    @staticmethod
    def _position_stats(match: SyntheticMatch) -> list[float]:
        """Summarise the simulated positions in a way ULP noise cannot move.

        Mean and standard deviation pin where the players and ball are; the mean
        absolute frame-to-frame change pins how they move, which a change to the
        step size or the pursuit rule would shift immediately while leaving the
        first two nearly intact.
        """
        tracking = match.tracking
        out: list[float] = []
        for array in (tracking.home_xy, tracking.away_xy, tracking.ball_xy):
            out.append(float(np.nanmean(array)))
            out.append(float(np.nanstd(array)))
            out.append(float(np.nanmean(np.abs(np.diff(array, axis=0)))))
        return out

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
        3: "648ff4ee6c71693e19c3dcb58d28f7906d6bf843e8871ac9404febe1436680df",
        7: "ed9fef36736602ed8aa1a64127a576ad204e9ae191884dd2d065adb772313bea",
        17: "80c47550287aa82d40613ee00154d9096d2fc61430b1704122197ab7dfb9ba0a",
        23: "eefb32498dd04c379ce322247b09ec7341e18701c0de8576caa1c810bae951a2",
    }

    #: Position summaries for the same seeds, verified identical to within
    #: 2e-16 relative on darwin/arm64 and linux/amd64. The comparison below
    #: allows 1e-9, which is seven orders of magnitude above the observed
    #: spread and far below any change a person would make on purpose.
    GOLDEN_STATS: ClassVar[dict[int, list[float]]] = {
        3: [
            -1.5029143307965462,
            18.493539722232292,
            0.07442660226746117,
            -1.1912559019858093,
            18.504199978698157,
            0.07777388551879111,
            -3.689134727015577,
            14.938456922888632,
            0.15607918807878854,
        ],
        7: [
            -0.5853952540072361,
            18.661640806640904,
            0.07939842304050683,
            -1.0108324101161152,
            18.471401079646988,
            0.07802442899415378,
            -2.7526884158491423,
            14.643071321361287,
            0.15902318636079318,
        ],
        17: [
            1.5700974745065224,
            19.248275396160892,
            0.07228343344902555,
            1.7525994616108727,
            18.136175368644334,
            0.0693965405226305,
            2.7183719643823667,
            13.64987137972367,
            0.14341560316538418,
        ],
        23: [
            -0.776718963741653,
            19.51521502038294,
            0.06461031056853506,
            -0.45958472030882513,
            17.737733517605655,
            0.06343224816842673,
            -2.064811367399106,
            12.803445779399256,
            0.12475946804263628,
        ],
    }

    @pytest.mark.parametrize("seed", [3, 7, 17, 23])
    def test_the_same_seed_gives_identical_output(self, seed: int) -> None:
        """Two calls in one process agree down to the last bit, positions included."""
        first = generate_synthetic_match(seed=seed, period_duration_s=45.0)
        second = generate_synthetic_match(seed=seed, period_duration_s=45.0)
        assert self._fingerprint(first) == self._fingerprint(second)

    @pytest.mark.parametrize("seed", [3, 7, 17, 23])
    def test_output_matches_the_recorded_fingerprint(self, seed: int) -> None:
        """The generator still produces exactly what the fixtures were built on."""
        match = generate_synthetic_match(seed=seed, period_duration_s=45.0)
        assert self._stable_fingerprint(match) == self.GOLDEN[seed]

    @pytest.mark.parametrize("seed", [3, 7, 17, 23])
    def test_simulated_positions_match_recorded_statistics(self, seed: int) -> None:
        """The simulation still puts players and the ball where it used to."""
        match = generate_synthetic_match(seed=seed, period_duration_s=45.0)
        assert self._position_stats(match) == approx(self.GOLDEN_STATS[seed], rel=1e-9)

    def test_different_seeds_diverge(self) -> None:
        a = generate_synthetic_match(seed=1, period_duration_s=45.0)
        b = generate_synthetic_match(seed=2, period_duration_s=45.0)
        assert self._fingerprint(a) != self._fingerprint(b)
        assert self._stable_fingerprint(a) != self._stable_fingerprint(b)

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
