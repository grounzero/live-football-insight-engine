"""Reader for the Metrica CSV format (sample games 1 and 2).

The layout is unusual enough to be worth describing, because the parser's shape
follows from it. Each team's tracking file has a three-row header::

    ,,,Home,,Home,,Home,,...          <- team label, one per player, then a gap
    ,,,11,,1,,2,,...                  <- shirt numbers
    Period,Frame,Time [s],Player11,,Player1,,...,Ball,

Every player occupies *two* columns (x and y) but is named only in the first,
so the column count is ``3 + 2 * n_players + 2``. Positions are in the unit
square with ``(0, 0)`` at the top left, and absent players are ``NaN``.

The ball appears in both team files. Its values agree, so the home file is
treated as authoritative and the away copy is used only as a cross-check —
disagreement means the two files are not from the same match.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from football_insights.domain import Event, EventType, MatchTracking, PlayerRef, Team
from football_insights.errors import DataValidationError
from football_insights.pitch import DEFAULT_PITCH, Pitch

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Metrica's sample data is sampled at 25 Hz.
DEFAULT_FRAME_RATE = 25.0

#: Largest ball-position difference tolerated between the two team files, in
#: source units. Both files carry the ball, so a real mismatch means they are
#: from different matches; this tolerance only absorbs float round-tripping.
BALL_AGREEMENT_TOLERANCE = 1e-6

#: Source event names mapped onto the canonical taxonomy. Anything unrecognised
#: becomes ``OTHER`` rather than being dropped, so an unexpected type shows up
#: in the report instead of vanishing.
EVENT_TYPE_MAP = {
    "PASS": EventType.PASS,
    "CARRY": EventType.CARRY,
    "SHOT": EventType.SHOT,
    "RECOVERY": EventType.RECOVERY,
    "BALL LOST": EventType.BALL_LOST,
    "CHALLENGE": EventType.CHALLENGE,
    "SET PIECE": EventType.SET_PIECE,
    "BALL OUT": EventType.BALL_OUT,
    "FAULT RECEIVED": EventType.FAULT_RECEIVED,
    "CARD": EventType.CARD,
}


def _to_float(value: str) -> float:
    """Parse a coordinate, treating blanks and ``NaN`` alike as missing."""
    text = value.strip()
    if not text or text.upper() == "NAN":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_team_file(
    path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse one team's tracking file.

    Args:
        path: Path to the CSV.

    Returns:
        Tuple of ``(shirt numbers, period, frame, time, positions)`` where
        positions has shape ``(n_frames, n_players + 1, 2)``; the final player
        slot is the ball.

    Raises:
        DataValidationError: If the header does not match the expected layout.
    """
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 4:
        msg = f"{path.name}: expected three header rows and data, found {len(rows)} rows"
        raise DataValidationError(msg)

    _, shirts = _parse_header(rows, path.name)
    period, frame, time_s, xy = _parse_frames(rows[3:], len(shirts), path.name)
    # The ball occupies the final object slot and has no shirt number.
    return shirts[:-1], period, frame, time_s, xy


def _parse_header(rows: list[list[str]], source: str) -> tuple[list[str], list[str]]:
    """Read the three-row Metrica header.

    Players are named in every second column from index 3, with their shirt
    number on the row above; the last named entry is the ball.

    Returns:
        ``(labels, shirt numbers)`` for every tracked object, ball last.

    Raises:
        DataValidationError: If the column layout is not the expected one.
    """
    shirt_row, name_row = rows[1], rows[2]
    if name_row[:3] != ["Period", "Frame", "Time [s]"]:
        msg = (
            f"{source}: third header row must start with "
            f"'Period,Frame,Time [s]', found {name_row[:3]}"
        )
        raise DataValidationError(msg)

    labels: list[str] = []
    shirts: list[str] = []
    for col in range(3, len(name_row), 2):
        label = name_row[col].strip()
        if not label:
            continue
        labels.append(label)
        shirts.append(shirt_row[col].strip() if col < len(shirt_row) else "")

    if not labels or labels[-1].lower() != "ball":
        msg = f"{source}: expected the final tracked object to be 'Ball', found {labels[-1:]}"
        raise DataValidationError(msg)
    return labels, shirts


def _parse_frames(
    data: list[list[str]], n_objects: int, source: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the data rows into columnar arrays.

    A coordinate pair that runs off the end of a short row stays ``NaN``, which
    is how this format represents a player who is not on the pitch.

    Raises:
        DataValidationError: If a row is too short to carry period, frame and time.
    """
    n_frames = len(data)
    period = np.empty(n_frames, dtype=np.int16)
    frame = np.empty(n_frames, dtype=np.int64)
    time_s = np.empty(n_frames, dtype=np.float64)
    xy = np.full((n_frames, n_objects, 2), np.nan)

    for i, row in enumerate(data):
        if len(row) < 3:
            # +4: three header rows, and rows are reported 1-based.
            msg = f"{source}: row {i + 4} has {len(row)} columns, expected at least 3"
            raise DataValidationError(msg)
        period[i] = int(row[0])
        frame[i] = int(row[1])
        time_s[i] = float(row[2])
        _fill_positions(row, n_objects, xy[i])

    return period, frame, time_s, xy


def _fill_positions(row: list[str], n_objects: int, out: np.ndarray) -> None:
    """Fill one frame's ``(n_objects, 2)`` coordinates from a data row.

    Every object occupies two adjacent columns from index 3. A pair that runs
    off the end of a short row is left untouched, so it keeps the ``NaN`` the
    caller pre-filled — this format's representation of an absent player.
    """
    for j in range(n_objects):
        cx, cy = 3 + 2 * j, 4 + 2 * j
        if cy < len(row):
            out[j, 0] = _to_float(row[cx])
            out[j, 1] = _to_float(row[cy])


def read_events(path: Path, pitch: Pitch = DEFAULT_PITCH) -> tuple[Event, ...]:
    """Parse a Metrica events CSV into canonical events.

    Args:
        path: Path to the events file.
        pitch: Pitch dimensions used to convert coordinates.

    Returns:
        Events in source order.

    Raises:
        DataValidationError: If required columns are missing.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Team", "Type", "Period", "Start Frame", "Start Time [s]"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            msg = f"{path.name}: events file is missing columns {sorted(missing)}"
            raise DataValidationError(msg)
        rows = list(reader)

    parsed = (_event_from_row(row, pitch) for row in rows)
    return tuple(event for event in parsed if event is not None)


def _event_from_row(row: dict[str, Any], pitch: Pitch) -> Event | None:
    """Convert one events-CSV row to an :class:`Event`.

    The row is typed as ``Any``-valued because that is what ``csv.DictReader``
    yields: a value is a string when the column is present, ``None`` when the
    row is short. Each field below narrows it explicitly.

    Returns:
        The event, or ``None`` for a row whose team is neither home nor away —
        the file uses blank and other markers for non-team annotations.
    """
    team_label = _text(row, "Team").lower()
    if team_label not in ("home", "away"):
        return None

    raw_type = _text(row, "Type").upper()
    start_frame, end_frame = _event_frames(row)
    start_time, end_time = _event_times(row)

    return Event(
        team=Team.HOME if team_label == "home" else Team.AWAY,
        type=EVENT_TYPE_MAP.get(raw_type, EventType.OTHER),
        subtype=_text(row, "Subtype") or None,
        # Deliberately unguarded: a missing period is a broken file, and this
        # raised before the parsing was split out. It must keep raising.
        period=int(row["Period"]),
        start_frame=start_frame,
        end_frame=end_frame,
        start_time_s=start_time,
        end_time_s=end_time,
        from_player=_text(row, "From") or None,
        to_player=_text(row, "To") or None,
        start_xy=_event_xy(row.get("Start X"), row.get("Start Y"), pitch),
        end_xy=_event_xy(row.get("End X"), row.get("End Y"), pitch),
        raw_type=raw_type,
    )


def _text(row: dict[str, Any], key: str) -> str:
    """Trimmed text for a column, empty when the column is absent or null."""
    return str(row.get(key) or "").strip()


def _event_frames(row: dict[str, Any]) -> tuple[int, int]:
    """Start and end frame.

    An absent or non-positive end frame means the event resolves where it
    started, and the end can never precede the start.
    """
    start = int(float(row["Start Frame"] or 0))
    raw_end = float(row.get("End Frame") or 0)
    end = int(raw_end) if raw_end > 0 else start
    return start, max(end, start)


def _event_times(row: dict[str, Any]) -> tuple[float, float]:
    """Start and end time in seconds, with the same end-defaults-to-start rule."""
    start = float(row["Start Time [s]"] or 0.0)
    raw_end = _text(row, "End Time [s]")
    end = float(raw_end) if raw_end else start
    return start, max(end, start)


def _event_xy(x: str | None, y: str | None, pitch: Pitch) -> tuple[float, float] | None:
    """Convert an event coordinate pair to canonical metres, or ``None``."""
    if x is None or y is None:
        return None
    fx, fy = _to_float(x), _to_float(y)
    if not (np.isfinite(fx) and np.isfinite(fy)):
        return None
    canonical = pitch.to_canonical(np.array([fx, fy]))
    return (float(canonical[0]), float(canonical[1]))


def _check_frame_index_agrees(
    home: tuple[np.ndarray, np.ndarray, np.ndarray],
    away: tuple[np.ndarray, np.ndarray, np.ndarray],
    names: tuple[str, str],
) -> None:
    """The two team files must describe the same frames of the same match.

    Args:
        home: ``(period, frame, positions)`` from the home file.
        away: The same from the away file.
        names: File names, used in the error message.

    Raises:
        DataValidationError: If the frame counts or the frame index disagree.
    """
    home_period, home_frame, home_xy = home
    away_period, away_frame, away_xy = away
    home_name, away_name = names
    if home_xy.shape[0] != away_xy.shape[0]:
        msg = (
            f"team files disagree on frame count: {home_name} has "
            f"{home_xy.shape[0]}, {away_name} has {away_xy.shape[0]}"
        )
        raise DataValidationError(msg)
    if not np.array_equal(home_frame, away_frame) or not np.array_equal(home_period, away_period):
        msg = f"{home_name} and {away_name} do not share a frame index"
        raise DataValidationError(msg)


def _check_ball_agrees(
    home_ball: np.ndarray, away_ball: np.ndarray, names: tuple[str, str]
) -> None:
    """Both files carry the ball; the two copies must match.

    Compared only where both files report a finite position, so a frame missing
    the ball in one file is not read as a disagreement.

    Raises:
        DataValidationError: If any shared frame differs beyond rounding.
    """
    both = np.isfinite(home_ball).all(axis=1) & np.isfinite(away_ball).all(axis=1)
    if not both.any():
        return
    divergence = np.abs(home_ball[both] - away_ball[both]).max()
    if divergence > BALL_AGREEMENT_TOLERANCE:
        home_name, away_name = names
        msg = (
            f"ball positions differ between {home_name} and {away_name} "
            f"by up to {divergence:.6f} in source units; the files are not from "
            "the same match"
        )
        raise DataValidationError(msg)


def read_match(
    home_path: Path,
    away_path: Path,
    events_path: Path,
    frame_rate: float = DEFAULT_FRAME_RATE,
    pitch: Pitch = DEFAULT_PITCH,
) -> tuple[MatchTracking, tuple[Event, ...]]:
    """Read a complete match from the three Metrica CSV files.

    Args:
        home_path: Home team tracking file.
        away_path: Away team tracking file.
        events_path: Events file.
        frame_rate: Tracking sample rate in hertz.
        pitch: Pitch dimensions used to convert coordinates.

    Returns:
        The tracking data in canonical coordinates and the parsed events.

    Raises:
        DataValidationError: If the two team files disagree.
    """
    home_shirts, period, frame, time_s, home_xy = _parse_team_file(home_path)
    away_shirts, away_period, away_frame, _, away_xy = _parse_team_file(away_path)

    names = (home_path.name, away_path.name)
    _check_frame_index_agrees((period, frame, home_xy), (away_period, away_frame, away_xy), names)
    # The ball is the final column of each file; the two copies must agree.
    home_ball = home_xy[:, -1, :]
    _check_ball_agrees(home_ball, away_xy[:, -1, :], names)

    tracking = MatchTracking(
        period=period,
        frame=frame,
        time_s=time_s,
        home_xy=pitch.to_canonical(home_xy[:, :-1, :]),
        away_xy=pitch.to_canonical(away_xy[:, :-1, :]),
        ball_xy=pitch.to_canonical(home_ball),
        home_players=_player_refs(home_shirts, Team.HOME),
        away_players=_player_refs(away_shirts, Team.AWAY),
        frame_rate=frame_rate,
    )
    return tracking, read_events(events_path, pitch)


def _player_refs(shirts: Sequence[str], team: Team) -> tuple[PlayerRef, ...]:
    """Build player references.

    The CSV format carries no position information, so goalkeepers are not
    identified here. :mod:`football_insights.data.orientation` infers them, and
    records that they were inferred rather than declared.
    """
    refs: list[PlayerRef] = []
    for shirt in shirts:
        number: int | None
        try:
            number = int(shirt)
        except (TypeError, ValueError):
            number = None
        refs.append(
            PlayerRef(
                player_id=f"{team.value}_{shirt or len(refs) + 1}",
                team=team,
                shirt_number=number,
                position_type=None,
                is_goalkeeper=False,
                goalkeeper_source="unknown",
            )
        )
    return tuple(refs)
