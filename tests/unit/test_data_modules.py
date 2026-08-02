"""Contracts of the four ingestion modules, pinned before a code-health pass.

`orientation`, `metrica_epts`, `metrica_csv` and `validate` were restructured
for maintainability. These tests describe what each module must keep doing —
parsing rules, dtypes, ordering, warning text and, importantly, which inputs are
*rejected*. A refactor that quietly starts accepting bad data is the failure mode
worth guarding against, so the error cases outnumber the happy ones.

Everything here builds its own fixtures. The real Metrica data is not committed,
and equivalence against it was verified separately (see docs/code-quality.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_insights.data import metrica_csv, metrica_epts, orientation, validate
from football_insights.domain import (
    AttackDirection,
    Event,
    EventType,
    MatchTracking,
    PlayerRef,
    Team,
)
from football_insights.errors import DataValidationError, OrientationError

# ---------------------------------------------------------------- fixtures


def _players(team: Team, count: int = 3) -> tuple[PlayerRef, ...]:
    return tuple(
        PlayerRef(
            player_id=f"{team.value}_{i}",
            team=team,
            shirt_number=i,
            position_type=None,
            is_goalkeeper=False,
            goalkeeper_source="unknown",
        )
        for i in range(count)
    )


def _tracking(
    *,
    n_frames: int = 400,
    periods: np.ndarray | None = None,
    frames: np.ndarray | None = None,
    times: np.ndarray | None = None,
    home_x: float = -30.0,
    away_x: float = 30.0,
    ball: np.ndarray | None = None,
    n_players: int = 3,
) -> MatchTracking:
    """A minimal but structurally valid match.

    Home sits at ``home_x`` and away at ``away_x`` for the whole period, which
    is enough for the centroid and goalkeeper signals to have a clear sign.
    """
    period = np.ones(n_frames, dtype=np.int16) if periods is None else periods
    frame = np.arange(1, n_frames + 1, dtype=np.int64) if frames is None else frames
    time_s = frame.astype(np.float64) / 25.0 if times is None else times
    home_xy = np.zeros((n_frames, n_players, 2))
    away_xy = np.zeros((n_frames, n_players, 2))
    home_xy[:, :, 0] = home_x
    away_xy[:, :, 0] = away_x
    ball_xy = np.zeros((n_frames, 2)) if ball is None else ball
    return MatchTracking(
        period=period,
        frame=frame,
        time_s=time_s,
        home_xy=home_xy,
        away_xy=away_xy,
        ball_xy=ball_xy,
        home_players=_players(Team.HOME, n_players),
        away_players=_players(Team.AWAY, n_players),
        frame_rate=25.0,
    )


def _pass(team: Team, period: int, frame: int, dx: float) -> Event:
    """A pass that progresses ``dx`` metres along x."""
    return Event(
        team=team,
        type=EventType.PASS,
        subtype=None,
        period=period,
        start_frame=frame,
        end_frame=frame,
        start_time_s=frame / 25.0,
        end_time_s=frame / 25.0,
        start_xy=(0.0, 0.0),
        end_xy=(dx, 0.0),
        raw_type="PASS",
    )


def _passes(team: Team, period: int, dx: float, count: int = 40) -> list[Event]:
    return [_pass(team, period, i + 1, dx) for i in range(count)]


# ---------------------------------------------------------------- orientation


class TestOrientationEvidence:
    """The tiered vote, and the structural facts it may never violate."""

    def test_declared_metadata_wins(self) -> None:
        tracking = _tracking()
        events = tuple(_passes(Team.HOME, 1, 2.0) + _passes(Team.AWAY, 1, -2.0))
        declared = {
            (1, Team.HOME): AttackDirection.POSITIVE_X,
            (1, Team.AWAY): AttackDirection.NEGATIVE_X,
        }
        result, _, _ = orientation.infer_orientation(tracking, events, "m", declared=declared)
        assert result.direction(1, Team.HOME) is AttackDirection.POSITIVE_X
        decision = result.report["decisions"][0]
        assert decision["source"] == "metadata"
        assert decision["signals"][0]["name"] == "provider_metadata"

    def test_direction_is_inferred_without_metadata(self) -> None:
        tracking = _tracking()
        events = tuple(_passes(Team.HOME, 1, 2.0) + _passes(Team.AWAY, 1, -2.0))
        result, _, _ = orientation.infer_orientation(tracking, events, "m")
        assert result.direction(1, Team.HOME) is AttackDirection.POSITIVE_X
        assert result.direction(1, Team.AWAY) is AttackDirection.NEGATIVE_X
        assert result.report["decisions"][0]["source"] == "inferred"

    def test_centroid_sign_is_away_from_own_goal(self) -> None:
        """A team camped at ``-x`` defends that end, so it attacks ``+x``.

        This is the sign error the project's orientation audit originally
        caught; getting it backwards inverts every spatial feature.
        """
        tracking = _tracking(home_x=-30.0, away_x=30.0)
        # No events at all, so only the position-based signals can vote.
        result, _, _ = orientation.infer_orientation(tracking, (), "m")
        assert result.direction(1, Team.HOME) is AttackDirection.POSITIVE_X
        assert result.direction(1, Team.AWAY) is AttackDirection.NEGATIVE_X

    def test_teams_swap_ends_in_the_second_half(self) -> None:
        n = 400
        periods = np.concatenate([np.ones(n, dtype=np.int16), np.full(n, 2, dtype=np.int16)])
        frames = np.arange(1, 2 * n + 1, dtype=np.int64)
        home_xy = np.zeros((2 * n, 3, 2))
        away_xy = np.zeros((2 * n, 3, 2))
        home_xy[:n, :, 0], away_xy[:n, :, 0] = -30.0, 30.0
        home_xy[n:, :, 0], away_xy[n:, :, 0] = 30.0, -30.0
        tracking = MatchTracking(
            period=periods,
            frame=frames,
            time_s=frames.astype(np.float64) / 25.0,
            home_xy=home_xy,
            away_xy=away_xy,
            ball_xy=np.zeros((2 * n, 2)),
            home_players=_players(Team.HOME),
            away_players=_players(Team.AWAY),
            frame_rate=25.0,
        )
        result, _, _ = orientation.infer_orientation(tracking, (), "m")
        assert result.direction(1, Team.HOME) is AttackDirection.POSITIVE_X
        assert result.direction(2, Team.HOME) is AttackDirection.NEGATIVE_X

    def test_failing_to_swap_ends_is_rejected(self) -> None:
        """A team attacking the same way in both halves is impossible."""
        n = 400
        periods = np.concatenate([np.ones(n, dtype=np.int16), np.full(n, 2, dtype=np.int16)])
        frames = np.arange(1, 2 * n + 1, dtype=np.int64)
        home_xy = np.zeros((2 * n, 3, 2))
        away_xy = np.zeros((2 * n, 3, 2))
        home_xy[:, :, 0], away_xy[:, :, 0] = -30.0, 30.0
        tracking = MatchTracking(
            period=periods,
            frame=frames,
            time_s=frames.astype(np.float64) / 25.0,
            home_xy=home_xy,
            away_xy=away_xy,
            ball_xy=np.zeros((2 * n, 2)),
            home_players=_players(Team.HOME),
            away_players=_players(Team.AWAY),
            frame_rate=25.0,
        )
        with pytest.raises(OrientationError, match="change ends at half time"):
            orientation.infer_orientation(tracking, (), "m")

    def test_both_teams_attacking_one_end_is_rejected(self) -> None:
        tracking = _tracking(home_x=-30.0, away_x=-30.0)
        with pytest.raises(OrientationError, match="cannot attack the same goal"):
            orientation.infer_orientation(tracking, (), "m")

    def test_no_evidence_names_the_override_key(self) -> None:
        """With nothing to go on, the error must say how to proceed."""
        tracking = _tracking(n_frames=10, home_x=np.nan, away_x=np.nan)
        with pytest.raises(OrientationError, match=r"m:1:home"):
            orientation.infer_orientation(tracking, (), "m")

    def test_an_override_is_recorded_with_its_reason(self) -> None:
        tracking = _tracking(n_frames=10, home_x=np.nan, away_x=np.nan)
        result, _, _ = orientation.infer_orientation(
            tracking,
            (),
            "m",
            overrides={"m:1:home": "+x", "m:1:away": "-x"},
            override_reasons={"m:1:home": "operator checked the footage"},
        )
        assert result.direction(1, Team.HOME) is AttackDirection.POSITIVE_X
        decision = result.report["decisions"][0]
        assert decision["source"] == "override"
        assert decision["override_reason"] == "operator checked the footage"

    def test_metadata_contradicted_by_high_volume_evidence_is_rejected(self) -> None:
        """Declared direction disagreeing with the play means mismatched files."""
        tracking = _tracking(home_x=-30.0, away_x=30.0)
        events = tuple(_passes(Team.HOME, 1, 5.0) + _passes(Team.AWAY, 1, -5.0))
        declared = {
            (1, Team.HOME): AttackDirection.NEGATIVE_X,
            (1, Team.AWAY): AttackDirection.POSITIVE_X,
        }
        with pytest.raises(OrientationError, match="contradictory evidence"):
            orientation.infer_orientation(tracking, events, "m", declared=declared)

    def test_the_report_lists_every_signal(self) -> None:
        tracking = _tracking()
        events = tuple(_passes(Team.HOME, 1, 2.0) + _passes(Team.AWAY, 1, -2.0))
        result, _, _ = orientation.infer_orientation(tracking, events, "m")
        names = {s["name"] for d in result.report["decisions"] for s in d["signals"]}
        assert "pass_progression" in names
        assert "team_centroid" in names
        assert result.report["min_agreement"] == orientation.MIN_AGREEMENT

    def test_goalkeepers_are_marked_and_sourced(self) -> None:
        tracking = _tracking()
        _, home, away = orientation.infer_orientation(tracking, (), "m")
        assert sum(p.is_goalkeeper for p in home) == 1
        assert sum(p.is_goalkeeper for p in away) == 1
        keeper = next(p for p in home if p.is_goalkeeper)
        assert keeper.goalkeeper_source == "inferred"


# ---------------------------------------------------------------- EPTS


def _metadata_xml(
    *,
    frame_rate: str = "25.0",
    teams: int = 2,
    players: int = 2,
    channel_refs: int | None = None,
    unknown_channel: bool = False,
    frame_range: bool = True,
    periods: bool = True,
    two_halves: bool = False,
) -> str:
    """Build a minimal EPTS metadata document, optionally malformed."""
    team_nodes = "".join(
        f'<Team id="T{t}"><ProviderParameters><ProviderParameter>'
        f"<Name>attack_direction_first_half</Name>"
        f"<Value>{'left_to_right' if t == 0 else 'right_to_left'}</Value>"
        f"</ProviderParameter></ProviderParameters></Team>"
        for t in range(teams)
    )
    player_nodes = "".join(
        f'<Player id="P{i}" teamId="T{i % max(teams, 1)}">'
        f"<ShirtNumber>{i + 1}</ShirtNumber>"
        f"<ProviderParameters><ProviderParameter><Name>position_type</Name>"
        f"<Value>{'Goalkeeper' if i == 0 else 'Defender'}</Value>"
        f"</ProviderParameter></ProviderParameters></Player>"
        for i in range(players)
    )
    channels = "".join(
        f'<PlayerChannel id="C{i}x" playerId="P{i}"/><PlayerChannel id="C{i}y" playerId="P{i}"/>'
        for i in range(players)
    )
    n_refs = channel_refs if channel_refs is not None else players * 2
    ref_ids = [f"C{i // 2}{'x' if i % 2 == 0 else 'y'}" for i in range(n_refs)]
    if unknown_channel:
        ref_ids[0] = "C-missing"
    refs = "".join(f'<PlayerChannelRef playerChannelId="{r}"/>' for r in ref_ids)
    span = ' startFrame="1" endFrame="1000"' if frame_range else ""
    period_params = ""
    if periods:
        period_params = (
            "<ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>"
            "<ProviderParameter><Name>first_half_end</Name><Value>1000</Value></ProviderParameter>"
        )
        if two_halves:
            period_params += (
                "<ProviderParameter><Name>second_half_start</Name>"
                "<Value>1200</Value></ProviderParameter>"
                "<ProviderParameter><Name>second_half_end</Name>"
                "<Value>2000</Value></ProviderParameter>"
            )
    return (
        "<Metadata>"
        f"<GlobalConfig><FrameRate>{frame_rate}</FrameRate>"
        f"<ProviderParameters>{period_params}</ProviderParameters></GlobalConfig>"
        f"<Teams>{team_nodes}</Teams><Players>{player_nodes}</Players>"
        f"<PlayerChannels>{channels}</PlayerChannels>"
        f"<DataFormatSpecification{span}>{refs}</DataFormatSpecification>"
        "</Metadata>"
    )


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


class TestEptsMetadata:
    """`read_metadata` interprets the XML, or refuses to."""

    def test_a_complete_document_parses(self, tmp_path: Path) -> None:
        meta = metrica_epts.read_metadata(_write(tmp_path, "m.xml", _metadata_xml()))
        assert meta.frame_rate == 25.0
        assert set(meta.team_sides.values()) == {Team.HOME, Team.AWAY}
        assert meta.declared_first_half[Team.HOME] is AttackDirection.POSITIVE_X
        assert meta.declared_first_half[Team.AWAY] is AttackDirection.NEGATIVE_X
        assert meta.periods == {1: (1, 1000)}
        assert len(meta.layouts) == 1
        assert meta.squad[0][2] == 1  # shirt number
        assert meta.squad[0][3] == "Goalkeeper"

    def test_the_first_team_listed_is_home(self, tmp_path: Path) -> None:
        meta = metrica_epts.read_metadata(_write(tmp_path, "m.xml", _metadata_xml()))
        assert meta.team_sides["T0"] is Team.HOME
        assert meta.team_sides["T1"] is Team.AWAY

    def test_missing_frame_rate_is_rejected(self, tmp_path: Path) -> None:
        xml = _metadata_xml().replace("<FrameRate>25.0</FrameRate>", "")
        with pytest.raises(DataValidationError, match="no FrameRate"):
            metrica_epts.read_metadata(_write(tmp_path, "m.xml", xml))

    def test_wrong_team_count_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="expected exactly 2 teams"):
            metrica_epts.read_metadata(_write(tmp_path, "m.xml", _metadata_xml(teams=1)))

    def test_no_players_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="no players found"):
            metrica_epts.read_metadata(
                _write(tmp_path, "m.xml", _metadata_xml(players=0, channel_refs=0))
            )

    def test_odd_channel_refs_are_rejected(self, tmp_path: Path) -> None:
        """X and y come in pairs; an odd count means the layout is unreadable."""
        with pytest.raises(DataValidationError, match="odd number of player"):
            metrica_epts.read_metadata(_write(tmp_path, "m.xml", _metadata_xml(channel_refs=3)))

    def test_unresolved_channel_reference_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="unresolved player channel"):
            metrica_epts.read_metadata(
                _write(tmp_path, "m.xml", _metadata_xml(unknown_channel=True))
            )

    def test_layout_without_a_frame_range_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="without a frame range"):
            metrica_epts.read_metadata(_write(tmp_path, "m.xml", _metadata_xml(frame_range=False)))

    def test_missing_periods_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="no period boundaries"):
            metrica_epts.read_metadata(_write(tmp_path, "m.xml", _metadata_xml(periods=False)))

    def test_declared_directions_flip_in_the_second_half(self, tmp_path: Path) -> None:
        """Teams change ends, so period 2 is the inverse of what was declared."""
        meta = metrica_epts.read_metadata(_write(tmp_path, "m.xml", _metadata_xml(two_halves=True)))
        assert meta.periods == {1: (1, 1000), 2: (1200, 2000)}
        declared = metrica_epts.declared_directions(meta)
        assert declared[(1, Team.HOME)] is AttackDirection.POSITIVE_X
        assert declared[(2, Team.HOME)] is AttackDirection.NEGATIVE_X
        assert declared[(1, Team.AWAY)] is AttackDirection.NEGATIVE_X
        assert declared[(2, Team.AWAY)] is AttackDirection.POSITIVE_X


class TestEptsTracking:
    """`read_tracking` scatters variable columns into stable per-squad slots."""

    @staticmethod
    def _meta(tmp_path: Path) -> metrica_epts.EptsMetadata:
        return metrica_epts.read_metadata(_write(tmp_path, "m.xml", _metadata_xml()))

    def test_frames_parse_with_the_documented_dtypes(self, tmp_path: Path) -> None:
        meta = self._meta(tmp_path)
        lines = "\n".join(f"{f}:0.5,0.5;0.6,0.6:0.5,0.5" for f in (1, 2, 3))
        tracking = metrica_epts.read_tracking(_write(tmp_path, "t.txt", lines), meta)
        assert tracking.n_frames == 3
        assert tracking.period.dtype == np.int16
        assert tracking.frame.dtype == np.int64
        assert tracking.time_s.dtype == np.float64
        assert tracking.home_xy.shape == (3, 1, 2)
        assert tracking.ball_xy.shape == (3, 2)
        np.testing.assert_array_equal(tracking.frame, [1, 2, 3])

    def test_frames_outside_the_periods_are_dropped(self, tmp_path: Path) -> None:
        """Pre-kickoff footage must not be assigned to a period."""
        meta = self._meta(tmp_path)
        lines = "\n".join(f"{f}:0.5,0.5;0.6,0.6:0.5,0.5" for f in (1, 2, 5000))
        tracking = metrica_epts.read_tracking(_write(tmp_path, "t.txt", lines), meta)
        assert tracking.n_frames == 2

    def test_no_usable_frames_is_rejected(self, tmp_path: Path) -> None:
        meta = self._meta(tmp_path)
        with pytest.raises(DataValidationError, match="no frames fell inside"):
            metrica_epts.read_tracking(
                _write(tmp_path, "t.txt", "5000:0.5,0.5;0.6,0.6:0.5,0.5"), meta
            )

    def test_a_short_line_is_rejected(self, tmp_path: Path) -> None:
        meta = self._meta(tmp_path)
        with pytest.raises(DataValidationError, match="colon-separated fields"):
            metrica_epts.read_tracking(_write(tmp_path, "t.txt", "1:0.5,0.5"), meta)

    def test_a_column_count_mismatch_is_rejected(self, tmp_path: Path) -> None:
        meta = self._meta(tmp_path)
        with pytest.raises(DataValidationError, match="player channels but its layout"):
            metrica_epts.read_tracking(_write(tmp_path, "t.txt", "1:0.5,0.5:0.5,0.5"), meta)

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        meta = self._meta(tmp_path)
        lines = "1:0.5,0.5;0.6,0.6:0.5,0.5\n\n2:0.5,0.5;0.6,0.6:0.5,0.5\n"
        tracking = metrica_epts.read_tracking(_write(tmp_path, "t.txt", lines), meta)
        assert tracking.n_frames == 2

    def test_an_absent_ball_becomes_nan(self, tmp_path: Path) -> None:
        meta = self._meta(tmp_path)
        tracking = metrica_epts.read_tracking(
            _write(tmp_path, "t.txt", "1:0.5,0.5;0.6,0.6:,"), meta
        )
        assert bool(np.isnan(tracking.ball_xy[0]).all())

    def test_repeated_parses_agree(self, tmp_path: Path) -> None:
        meta = self._meta(tmp_path)
        path = _write(tmp_path, "t.txt", "1:0.5,0.5;0.6,0.6:0.5,0.5\n2:0.4,0.4;0.3,0.3:0.4,0.4")
        first = metrica_epts.read_tracking(path, meta)
        second = metrica_epts.read_tracking(path, meta)
        np.testing.assert_array_equal(first.home_xy, second.home_xy)
        np.testing.assert_array_equal(first.ball_xy, second.ball_xy)


# ---------------------------------------------------------------- CSV


def _team_csv(
    *,
    rows: int = 3,
    third_header: str = "Period,Frame,Time [s]",
    last_object: str = "Ball",
    short_row: bool = False,
    ball_x: str = "0.5",
) -> str:
    header = f",,,Home,,Home,,\n,,,11,,1,,\n{third_header},Player11,,{last_object},\n"
    body = ""
    for i in range(1, rows + 1):
        if short_row:
            body += "1,\n"
        else:
            body += f"1,{i},{i / 25.0},0.4,0.4,{ball_x},0.5\n"
    return header + body


def _events_csv(*, team: str = "Home", period: str = "1", drop_column: str = "") -> str:
    columns = [
        "Team",
        "Type",
        "Subtype",
        "Period",
        "Start Frame",
        "Start Time [s]",
        "End Frame",
        "End Time [s]",
        "From",
        "To",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
    ]
    values = [
        team,
        "PASS",
        "",
        period,
        "1",
        "0.04",
        "3",
        "0.12",
        "Player1",
        "Player2",
        "0.4",
        "0.4",
        "0.6",
        "0.6",
    ]
    if drop_column:
        index = columns.index(drop_column)
        columns.pop(index)
        values.pop(index)
    return ",".join(columns) + "\n" + ",".join(values) + "\n"


def _read_csv_match(tmp_path: Path, home: str, away: str | None = None) -> MatchTracking:
    """Drive a pair of team files through the public reader."""
    tracking, _ = metrica_csv.read_match(
        _write(tmp_path, "home.csv", home),
        _write(tmp_path, "away.csv", away if away is not None else home),
        _write(tmp_path, "e.csv", _events_csv()),
    )
    return tracking


class TestCsvTracking:
    """The three-row Metrica header, read through the public entry point."""

    def test_a_valid_file_parses_with_expected_dtypes(self, tmp_path: Path) -> None:
        tracking = _read_csv_match(tmp_path, _team_csv())
        assert tracking.period.dtype == np.int16
        assert tracking.frame.dtype == np.int64
        assert tracking.time_s.dtype == np.float64
        assert tracking.home_xy.shape == (3, 1, 2)
        assert tracking.ball_xy.shape == (3, 2)
        assert [p.shirt_number for p in tracking.home_players] == [11]
        np.testing.assert_array_equal(tracking.frame, [1, 2, 3])

    def test_a_wrong_third_header_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="third header row must start"):
            _read_csv_match(tmp_path, _team_csv(third_header="Half,Frame,Time"))

    def test_a_missing_ball_column_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="final tracked object to be 'Ball'"):
            _read_csv_match(tmp_path, _team_csv(last_object="Player7"))

    def test_a_file_without_data_rows_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="three header rows and data"):
            _read_csv_match(tmp_path, _team_csv(rows=0))

    def test_a_short_data_row_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="expected at least 3"):
            _read_csv_match(tmp_path, _team_csv(short_row=True))

    def test_blank_and_invalid_coordinates_become_nan(self, tmp_path: Path) -> None:
        """A missing position must never be silently coerced to a number.

        Only the ball's x is blanked here, so the y beside it must survive: a
        parser that dropped the whole pair would also pass a looser assertion.
        """
        blank = _read_csv_match(tmp_path, _team_csv(ball_x=""))
        assert bool(np.isnan(blank.ball_xy[0, 0]))
        assert np.isfinite(blank.ball_xy[0, 1])

        junk = _read_csv_match(tmp_path, _team_csv(ball_x="not-a-number"))
        assert bool(np.isnan(junk.ball_xy[0, 0]))
        assert np.isfinite(junk.ball_xy[0, 1])

    def test_repeated_parses_agree(self, tmp_path: Path) -> None:
        first = _read_csv_match(tmp_path, _team_csv())
        second = _read_csv_match(tmp_path, _team_csv())
        np.testing.assert_array_equal(first.home_xy, second.home_xy)
        np.testing.assert_array_equal(first.ball_xy, second.ball_xy)


class TestCsvEvents:
    """`read_events` converts rows, skipping non-team annotations."""

    def test_a_valid_row_becomes_an_event(self, tmp_path: Path) -> None:
        (event,) = metrica_csv.read_events(_write(tmp_path, "e.csv", _events_csv()))
        assert event.team is Team.HOME
        assert event.type is EventType.PASS
        assert event.period == 1
        assert event.start_frame == 1
        assert event.end_frame == 3
        assert event.from_player == "Player1"
        assert event.start_xy is not None

    def test_rows_for_other_teams_are_skipped(self, tmp_path: Path) -> None:
        assert metrica_csv.read_events(_write(tmp_path, "e.csv", _events_csv(team="Referee"))) == ()

    def test_missing_required_columns_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="missing columns"):
            metrica_csv.read_events(
                _write(tmp_path, "e.csv", _events_csv(drop_column="Start Frame"))
            )

    def test_a_missing_period_still_raises(self, tmp_path: Path) -> None:
        """A broken period is a broken file, not something to default away."""
        with pytest.raises(ValueError, match="invalid literal"):
            metrica_csv.read_events(_write(tmp_path, "e.csv", _events_csv(period="")))


class TestCsvReadMatch:
    """The two team files must describe the same match."""

    def test_a_frame_count_mismatch_is_rejected(self, tmp_path: Path) -> None:
        home = _write(tmp_path, "home.csv", _team_csv(rows=3))
        away = _write(tmp_path, "away.csv", _team_csv(rows=4))
        events = _write(tmp_path, "e.csv", _events_csv())
        with pytest.raises(DataValidationError, match="disagree on frame count"):
            metrica_csv.read_match(home, away, events)

    def test_a_ball_disagreement_is_rejected(self, tmp_path: Path) -> None:
        home = _write(tmp_path, "home.csv", _team_csv(ball_x="0.5"))
        away = _write(tmp_path, "away.csv", _team_csv(ball_x="0.9"))
        events = _write(tmp_path, "e.csv", _events_csv())
        with pytest.raises(DataValidationError, match="ball positions differ"):
            metrica_csv.read_match(home, away, events)

    def test_matching_files_produce_tracking(self, tmp_path: Path) -> None:
        home = _write(tmp_path, "home.csv", _team_csv())
        away = _write(tmp_path, "away.csv", _team_csv())
        events = _write(tmp_path, "e.csv", _events_csv())
        tracking, parsed = metrica_csv.read_match(home, away, events)
        assert tracking.n_frames == 3
        assert tracking.home_xy.shape == (3, 1, 2)
        assert tracking.ball_xy.shape == (3, 2)
        assert len(parsed) == 1


# ---------------------------------------------------------------- validation


class TestValidateTracking:
    """Each rule independently, and the order findings arrive in."""

    def test_a_clean_match_reports_nothing(self) -> None:
        report = validate.validate_tracking(_tracking(), "m")
        assert report.warnings == []
        assert report.duplicate_frames == 0
        assert report.frame_gaps == 0

    def test_no_frames_is_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="contains no frames"):
            validate.validate_tracking(_tracking(n_frames=0), "m")

    def test_a_non_positive_frame_rate_is_rejected(self) -> None:
        tracking = _tracking()
        broken = MatchTracking(
            period=tracking.period,
            frame=tracking.frame,
            time_s=tracking.time_s,
            home_xy=tracking.home_xy,
            away_xy=tracking.away_xy,
            ball_xy=tracking.ball_xy,
            home_players=tracking.home_players,
            away_players=tracking.away_players,
            frame_rate=0.0,
        )
        with pytest.raises(DataValidationError, match="frame rate must be positive"):
            validate.validate_tracking(broken, "m")

    def test_backwards_periods_are_rejected(self) -> None:
        periods = np.ones(400, dtype=np.int16)
        periods[200:] = 0
        with pytest.raises(DataValidationError, match="periods are not monotonically"):
            validate.validate_tracking(_tracking(periods=periods), "m")

    def test_backwards_frames_are_rejected(self) -> None:
        frames = np.arange(1, 401, dtype=np.int64)
        frames[200] = 5
        with pytest.raises(DataValidationError, match="frames go backwards"):
            validate.validate_tracking(_tracking(frames=frames), "m")

    def test_backwards_timestamps_are_rejected(self) -> None:
        times = np.arange(400, dtype=np.float64) / 25.0
        times[300] = 0.0
        with pytest.raises(DataValidationError, match="timestamps are not monotonically"):
            validate.validate_tracking(_tracking(times=times), "m")

    def test_duplicate_frames_are_counted_not_fatal(self) -> None:
        frames = np.arange(1, 401, dtype=np.int64)
        frames[200] = frames[199]
        report = validate.validate_tracking(_tracking(frames=frames), "m")
        assert report.duplicate_frames == 1
        assert "duplicate frame indices" in report.warnings[0]

    def test_frame_gaps_are_counted_with_the_largest(self) -> None:
        frames = np.arange(1, 401, dtype=np.int64)
        frames[200:] += 10
        report = validate.validate_tracking(_tracking(frames=frames), "m")
        assert report.frame_gaps == 1
        assert report.largest_gap_frames == 10
        assert "gaps in the frame sequence" in report.warnings[0]

    def test_missing_ball_is_reported(self) -> None:
        ball = np.zeros((400, 2))
        ball[:100] = np.nan
        report = validate.validate_tracking(_tracking(ball=ball), "m")
        assert report.ball_missing == 100
        assert report.ball_missing_ratio == pytest.approx(0.25)
        assert "ball missing in 100 frames" in report.warnings[0]

    def test_in_play_ball_loss_beyond_the_limit_is_fatal(self) -> None:
        ball = np.zeros((400, 2))
        ball[:300] = np.nan
        in_play = np.ones(400, dtype=bool)
        with pytest.raises(DataValidationError, match="above the 40% limit"):
            validate.validate_tracking(_tracking(ball=ball), "m", in_play=in_play)

    def test_dead_ball_loss_is_tolerated(self) -> None:
        """The same missing frames are acceptable when play is stopped."""
        ball = np.zeros((400, 2))
        ball[:300] = np.nan
        in_play = np.zeros(400, dtype=bool)
        in_play[300:] = True
        report = validate.validate_tracking(_tracking(ball=ball), "m", in_play=in_play)
        assert report.ball_missing_ratio_in_play == 0.0

    def test_players_far_off_the_pitch_are_fatal(self) -> None:
        """Implausible coordinates mean the units are wrong, not that play was wide."""
        tracking = _tracking(home_x=5000.0, away_x=30.0)
        with pytest.raises(DataValidationError, match="coordinate system or"):
            validate.validate_tracking(tracking, "m")

    def test_a_ball_far_off_the_pitch_is_fatal(self) -> None:
        ball = np.full((400, 2), 5000.0)
        with pytest.raises(DataValidationError, match="check the coordinate convention"):
            validate.validate_tracking(_tracking(ball=ball), "m")

    def test_warning_order_is_stable(self) -> None:
        """Duplicates, then gaps, then ball coverage."""
        frames = np.arange(1, 401, dtype=np.int64)
        frames[100] = frames[99]
        frames[200:] += 10
        ball = np.zeros((400, 2))
        ball[:50] = np.nan
        report = validate.validate_tracking(_tracking(frames=frames, ball=ball), "m")
        assert "duplicate frame indices" in report.warnings[0]
        assert "gaps in the frame sequence" in report.warnings[1]
        assert "ball missing" in report.warnings[2]


class TestValidateEvents:
    """Events must align with the tracking they are paired with."""

    @staticmethod
    def _report() -> validate.ValidationReport:
        return validate.ValidationReport(
            match_id="m", n_frames=400, frame_rate=25.0, duration_s=16.0
        )

    def test_aligned_events_are_kept_and_sorted(self) -> None:
        tracking = _tracking()
        events = (_pass(Team.HOME, 1, 200, 1.0), _pass(Team.HOME, 1, 100, 1.0))
        kept = validate.validate_events(events, tracking, "m", self._report())
        assert [e.start_frame for e in kept] == [100, 200]

    def test_out_of_order_events_are_counted(self) -> None:
        tracking = _tracking()
        report = self._report()
        events = (_pass(Team.HOME, 1, 200, 1.0), _pass(Team.HOME, 1, 100, 1.0))
        validate.validate_events(events, tracking, "m", report)
        assert "arrived out of order" in report.warnings[-1]

    def test_no_events_is_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="no events parsed"):
            validate.validate_events((), _tracking(), "m", self._report())

    def test_events_outside_the_frame_range_are_dropped(self) -> None:
        tracking = _tracking()
        report = self._report()
        events = (_pass(Team.HOME, 1, 100, 1.0), _pass(Team.HOME, 1, 99_999, 1.0))
        kept = validate.validate_events(events, tracking, "m", report)
        assert len(kept) == 1
        assert report.events_dropped == 1
        assert "were dropped" in report.warnings[-1]

    def test_completely_unaligned_events_are_rejected(self) -> None:
        """Nothing aligning means the files are from different matches."""
        events = (_pass(Team.HOME, 1, 99_999, 1.0),)
        with pytest.raises(DataValidationError, match="different matches"):
            validate.validate_events(events, _tracking(), "m", self._report())
