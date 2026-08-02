"""Reader for the Metrica EPTS format (sample game 3).

Tracking is one line per frame::

    <frame>:<x>,<y>;<x>,<y>;...;<x>,<y>:<ball_x>,<ball_y>

Player channels appear in a fixed order given by the metadata XML, which also
supplies the frame rate, period boundaries, each player's ``position_type`` —
including which player is the goalkeeper — and, uniquely among the three sample
matches, an explicit ``attack_direction_first_half`` per team.

That last field is why this reader matters out of proportion to the one match it
serves: it is the only declared playing direction in the dataset, so game 3 acts
as ground truth for the inference applied to games 1 and 2.

The format also differs from the CSV in two ways worth handling explicitly: the
match does not start at frame 1 (there is pre-kickoff footage), and the event
stream includes ``CARRY`` events the CSV matches do not have.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from football_insights.domain import (
    AttackDirection,
    Event,
    EventType,
    MatchTracking,
    PlayerRef,
    Team,
)
from football_insights.errors import DataValidationError
from football_insights.pitch import DEFAULT_PITCH, Pitch
from football_insights.types import FloatArray

if TYPE_CHECKING:
    from collections.abc import Sequence

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


@dataclass(frozen=True, slots=True)
class ChannelLayout:
    """Column order in force over one range of frames.

    EPTS writes only the players currently on the pitch, so the meaning of
    column *j* changes at every substitution. Each ``DataFormatSpecification``
    block declares the layout for a frame range; there are eleven of them in
    sample game 3.
    """

    start_frame: int
    end_frame: int
    #: Player id per parsed column, in file order.
    player_ids: tuple[str, ...]

    def covers(self, frame: int) -> bool:
        """Whether this layout applies to a frame."""
        return self.start_frame <= frame <= self.end_frame


@dataclass(frozen=True, slots=True)
class EptsMetadata:
    """Everything the metadata XML tells us."""

    frame_rate: float
    #: Team id to canonical side. The first team listed is treated as home.
    team_sides: dict[str, Team]
    #: Full squads: ``(player_id, team, shirt_number, position_type)``, in the
    #: order they are declared. Every squad member gets a stable column for the
    #: whole match, so a substitute is ``NaN`` until they come on.
    squad: tuple[tuple[str, Team, int | None, str | None], ...]
    #: Column layouts, ordered by start frame.
    layouts: tuple[ChannelLayout, ...]
    #: Declared attacking direction for the first half, per team.
    declared_first_half: dict[Team, AttackDirection]
    #: Period boundaries as ``period -> (start_frame, end_frame)``.
    periods: dict[int, tuple[int, int]]

    def layout_for(self, frame: int) -> ChannelLayout | None:
        """The layout covering a frame, or ``None`` outside every range."""
        for layout in self.layouts:
            if layout.covers(frame):
                return layout
        return None


def _provider_parameters(node: ET.Element) -> dict[str, str]:
    """Collect ``ProviderParameter`` name/value pairs beneath a node."""
    out: dict[str, str] = {}
    for parameter in node.iter("ProviderParameter"):
        name = parameter.findtext("Name")
        value = parameter.findtext("Value")
        if name:
            out[name] = (value or "").strip()
    return out


def _parse_frame_rate(root: ET.Element, source: str) -> float:
    """Read the tracking sample rate."""
    frame_rate_text = root.findtext(".//FrameRate")
    if not frame_rate_text:
        msg = f"{source}: no FrameRate in metadata"
        raise DataValidationError(msg)
    return float(frame_rate_text)


def _parse_teams(
    root: ET.Element, source: str
) -> tuple[dict[str, Team], dict[Team, AttackDirection]]:
    """Map team ids to sides, and read any declared first-half direction.

    The first team listed is treated as home; EPTS does not say which is which,
    and this ordering is what the rest of the pipeline is calibrated against.
    """
    team_nodes = list(root.iter("Team"))
    if len(team_nodes) != 2:
        msg = f"{source}: expected exactly 2 teams, found {len(team_nodes)}"
        raise DataValidationError(msg)

    team_sides: dict[str, Team] = {}
    declared: dict[Team, AttackDirection] = {}
    for index, node in enumerate(team_nodes):
        team_id = node.get("id") or f"team{index}"
        side = Team.HOME if index == 0 else Team.AWAY
        team_sides[team_id] = side
        direction = _provider_parameters(node).get("attack_direction_first_half", "")
        if direction == "left_to_right":
            declared[side] = AttackDirection.POSITIVE_X
        elif direction == "right_to_left":
            declared[side] = AttackDirection.NEGATIVE_X
    return team_sides, declared


def _parse_squad(
    root: ET.Element, team_sides: dict[str, Team], source: str
) -> tuple[tuple[str, Team, int | None, str | None], ...]:
    """Read every declared player, in file order.

    Players belonging to an unknown team are skipped rather than rejected: the
    file lists officials and other non-squad entries under the same element.
    """
    squad: list[tuple[str, Team, int | None, str | None]] = []
    for player in root.iter("Player"):
        side = team_sides.get(player.get("teamId") or "")
        if side is None:
            continue
        shirt_text = player.findtext("ShirtNumber")
        shirt = int(shirt_text) if shirt_text and shirt_text.strip().isdigit() else None
        squad.append(
            (
                player.get("id") or "",
                side,
                shirt,
                _provider_parameters(player).get("position_type"),
            )
        )
    if not squad:
        msg = f"{source}: no players found"
        raise DataValidationError(msg)
    return tuple(squad)


def _channel_player_ids(
    spec: ET.Element, channel_player: dict[str | None, str | None]
) -> list[str | None]:
    """Resolve a spec's channel refs to player ids, in file order.

    An entry is ``None`` when the ref names a channel the metadata never
    declared, which makes the whole layout uninterpretable.
    """
    return [channel_player.get(r.get("playerChannelId")) for r in spec.iter("PlayerChannelRef")]


def _layout_frame_range(spec: ET.Element, source: str) -> tuple[int, int]:
    """Read the frame range a layout applies to."""
    start = spec.get("startFrame")
    end = spec.get("endFrame")
    if start is None or end is None:
        msg = f"{source}: DataFormatSpecification without a frame range"
        raise DataValidationError(msg)
    return int(start), int(end)


def _parse_layout(
    spec: ET.Element, channel_player: dict[str | None, str | None], source: str
) -> ChannelLayout:
    """Read one ``DataFormatSpecification`` block into a column layout.

    x and y are separate channels, so the refs come in pairs and every second
    one names the same player.
    """
    refs = _channel_player_ids(spec, channel_player)
    if len(refs) % 2 != 0:
        msg = (
            f"{source}: DataFormatSpecification has an odd number of player "
            f"channel refs ({len(refs)}); x and y must come in pairs"
        )
        raise DataValidationError(msg)

    players = refs[::2]
    if any(p is None for p in players):
        unknown = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        msg = f"{source}: unresolved player channel references in {unknown[:4]}"
        raise DataValidationError(msg)

    start, end = _layout_frame_range(spec, source)
    return ChannelLayout(
        start_frame=start,
        end_frame=end,
        player_ids=tuple(str(p) for p in players),
    )


def _parse_layouts(root: ET.Element, source: str) -> tuple[ChannelLayout, ...]:
    """Read every column layout, ordered by the frame range it applies to.

    EPTS writes only the players on the pitch, so the meaning of column *j*
    changes at each substitution and each range needs its own layout.
    """
    channel_player: dict[str | None, str | None] = {
        c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")
    }
    layouts = [
        _parse_layout(spec, channel_player, source)
        for spec in root.iter("DataFormatSpecification")
        if _channel_player_ids(spec, channel_player)
    ]
    if not layouts:
        msg = f"{source}: no DataFormatSpecification blocks found"
        raise DataValidationError(msg)
    layouts.sort(key=lambda item: item.start_frame)
    return tuple(layouts)


def _parse_periods(root: ET.Element, source: str) -> dict[int, tuple[int, int]]:
    """Read the period frame boundaries."""
    # An Element is falsy when it has no children, so `or` would silently pick
    # the wrong node; ElementTree deprecates truth-testing for exactly this reason.
    global_config = root.find(".//GlobalConfig")
    globals_ = _provider_parameters(global_config if global_config is not None else root)
    periods: dict[int, tuple[int, int]] = {}
    for number, (start_key, end_key) in enumerate(
        (
            ("first_half_start", "first_half_end"),
            ("second_half_start", "second_half_end"),
        ),
        start=1,
    ):
        start, end = globals_.get(start_key, ""), globals_.get(end_key, "")
        if start and end:
            periods[number] = (int(start), int(end))
    if not periods:
        msg = f"{source}: no period boundaries in metadata"
        raise DataValidationError(msg)
    return periods


def read_metadata(path: Path) -> EptsMetadata:
    """Parse the EPTS metadata XML.

    Args:
        path: Path to the metadata file.

    Returns:
        The parsed metadata.

    Raises:
        DataValidationError: If required fields are absent.
    """
    # ElementTree is not hardened against hostile XML (billion laughs, external
    # entities). The input is a local file the operator downloaded themselves
    # via `make data`, not untrusted network input.
    root = ET.parse(path).getroot()
    source = path.name
    team_sides, declared = _parse_teams(root, source)
    return EptsMetadata(
        frame_rate=_parse_frame_rate(root, source),
        team_sides=team_sides,
        squad=_parse_squad(root, team_sides, source),
        layouts=_parse_layouts(root, source),
        declared_first_half=declared,
        periods=_parse_periods(root, source),
    )


def _parse_pair(text: str) -> tuple[float, float]:
    """Parse ``x,y``, mapping missing values to ``NaN``."""
    if not text or text.upper().startswith("NAN"):
        return (float("nan"), float("nan"))
    parts = text.split(",")
    if len(parts) < 2:
        return (float("nan"), float("nan"))
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return (float("nan"), float("nan"))


@dataclass(frozen=True, slots=True)
class _LineRef:
    """Where a parsed line came from, carried only for error messages."""

    source: str
    line_number: int
    frame: int = -1

    def at_frame(self, frame: int) -> _LineRef:
        """The same reference, now knowing which frame the line describes."""
        return _LineRef(self.source, self.line_number, frame)


@dataclass(frozen=True, slots=True)
class _ScatterPlan:
    """Where each parsed column lands in the stable per-squad arrays.

    EPTS writes only the players currently on the pitch, so a frame's column
    order depends on which layout covers it. Resolving that once per layout
    keeps it out of the per-frame loop.
    """

    n_home: int
    n_away: int
    #: Layout index to per-column ``(is_home, column)`` destinations.
    columns: dict[int, list[tuple[bool, int]]]


def _build_scatter_plan(metadata: EptsMetadata) -> _ScatterPlan:
    """Resolve every layout's columns to stable per-squad positions."""
    home_squad = [p for p in metadata.squad if p[1] is Team.HOME]
    away_squad = [p for p in metadata.squad if p[1] is Team.AWAY]
    slot = {p[0]: (p[1], i) for i, p in enumerate(home_squad)}
    slot.update({p[0]: (p[1], i) for i, p in enumerate(away_squad)})
    return _ScatterPlan(
        n_home=len(home_squad),
        n_away=len(away_squad),
        columns={
            index: [
                (slot[player_id][0] is Team.HOME, slot[player_id][1])
                for player_id in layout.player_ids
            ]
            for index, layout in enumerate(metadata.layouts)
        },
    )


def _locate_frame(metadata: EptsMetadata, ref: _LineRef) -> tuple[int, int] | None:
    """Find the period and layout covering a frame.

    Returns:
        ``(period, layout_index)``, or ``None`` for frames outside every declared
        period — pre-kickoff footage and the half-time break, which are dropped
        rather than assigned to a period they do not belong to.

    Raises:
        DataValidationError: If the frame is in a period but no layout covers it,
            which would leave its columns uninterpretable.
    """
    period = next((p for p, (lo, hi) in metadata.periods.items() if lo <= ref.frame <= hi), None)
    if period is None:
        return None
    layout_index = next(
        (i for i, lay in enumerate(metadata.layouts) if lay.covers(ref.frame)), None
    )
    if layout_index is None:
        msg = (
            f"{ref.source}: frame {ref.frame} on line {ref.line_number} is not covered by "
            "any DataFormatSpecification range"
        )
        raise DataValidationError(msg)
    return period, layout_index


def _scatter_row(
    plan: _ScatterPlan, layout_index: int, fields: list[str], ref: _LineRef
) -> tuple[FloatArray, FloatArray]:
    """Scatter one frame's player channels into stable per-squad rows.

    A player not on the pitch stays ``NaN``, matching the CSV reader and the
    contract :class:`~football_insights.domain.Frame` relies on.

    Raises:
        DataValidationError: If the field count disagrees with the layout.
    """
    columns = plan.columns[layout_index]
    if len(fields) != len(columns):
        msg = (
            f"{ref.source}: line {ref.line_number} (frame {ref.frame}) has {len(fields)} "
            f"player channels but its layout declares {len(columns)}"
        )
        raise DataValidationError(msg)

    home_row: FloatArray = np.full((plan.n_home, 2), np.nan)
    away_row: FloatArray = np.full((plan.n_away, 2), np.nan)
    for field_text, (is_home, col) in zip(fields, columns, strict=True):
        target = home_row if is_home else away_row
        target[col] = _parse_pair(field_text)
    return home_row, away_row


@dataclass(slots=True)
class _TrackingRows:
    """Per-frame rows accumulated in file order."""

    periods: list[int] = field(default_factory=list[int])
    frames: list[int] = field(default_factory=list[int])
    home: list[FloatArray] = field(default_factory=list["FloatArray"])
    away: list[FloatArray] = field(default_factory=list["FloatArray"])
    balls: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])


def _read_tracking_rows(path: Path, metadata: EptsMetadata, plan: _ScatterPlan) -> _TrackingRows:
    """Read every usable frame from the tracking file.

    Raises:
        DataValidationError: On a structurally unreadable line.
    """
    rows = _TrackingRows()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(":")
            ref = _LineRef(path.name, line_number)
            if len(parts) < 3:
                msg = f"{ref.source}: line {line_number} has {len(parts)} colon-separated fields"
                raise DataValidationError(msg)

            ref = ref.at_frame(int(parts[0]))
            located = _locate_frame(metadata, ref)
            if located is None:
                continue
            period, layout_index = located

            home_row, away_row = _scatter_row(plan, layout_index, parts[1].split(";"), ref)
            rows.periods.append(period)
            rows.frames.append(ref.frame)
            rows.home.append(home_row)
            rows.away.append(away_row)
            rows.balls.append(_parse_pair(parts[2]))
    return rows


def read_tracking(
    path: Path,
    metadata: EptsMetadata,
    pitch: Pitch = DEFAULT_PITCH,
) -> MatchTracking:
    """Parse the EPTS tracking file.

    Two things need care. First, only the players on the pitch are written, and
    the column order changes at every substitution, so each frame is scattered
    into a **stable** per-squad column; a player not on the pitch is ``NaN``,
    matching the CSV format's behaviour and the contract
    :class:`~football_insights.domain.Frame` relies on.

    Second, frames outside the declared period boundaries — pre-kickoff footage
    and the half-time break — are discarded rather than assigned to a period
    they do not belong to.

    Args:
        path: Path to the tracking file.
        metadata: Parsed metadata.
        pitch: Pitch dimensions used to convert coordinates.

    Returns:
        Tracking in canonical coordinates.

    Raises:
        DataValidationError: If a frame's column count disagrees with the layout
            declared for it, or no frames fall inside the period boundaries.
    """
    plan = _build_scatter_plan(metadata)
    rows = _read_tracking_rows(path, metadata, plan)

    if not rows.frames:
        msg = f"{path.name}: no frames fell inside the declared period boundaries"
        raise DataValidationError(msg)

    times = np.array(rows.frames, dtype=np.float64) / metadata.frame_rate
    return MatchTracking(
        period=np.array(rows.periods, dtype=np.int16),
        frame=np.array(rows.frames, dtype=np.int64),
        time_s=times,
        home_xy=pitch.to_canonical(np.stack(rows.home)),
        away_xy=pitch.to_canonical(np.stack(rows.away)),
        ball_xy=pitch.to_canonical(np.array(rows.balls)),
        home_players=_player_refs(metadata.squad, Team.HOME),
        away_players=_player_refs(metadata.squad, Team.AWAY),
        frame_rate=metadata.frame_rate,
    )


def _player_refs(
    squad: Sequence[tuple[str, Team, int | None, str | None]], team: Team
) -> tuple[PlayerRef, ...]:
    """Build player references, honouring the declared goalkeeper.

    Unlike the CSV format, EPTS states each player's position, so goalkeepers
    are ``declared`` rather than inferred. The distinction is carried through to
    the direction report so it never implies the source said something it did not.
    """
    return tuple(
        PlayerRef(
            player_id=player_id,
            team=side,
            shirt_number=shirt,
            position_type=position,
            is_goalkeeper=(position or "").strip().lower() == "goalkeeper",
            goalkeeper_source="declared" if position else "unknown",
        )
        for player_id, side, shirt, position in squad
        if side is team
    )


def read_events(
    path: Path,
    metadata: EptsMetadata,
    pitch: Pitch = DEFAULT_PITCH,
) -> tuple[Event, ...]:
    """Parse the JSON event stream.

    Args:
        path: Path to the events file.
        metadata: Parsed metadata, used to map team ids to sides.
        pitch: Pitch dimensions used to convert coordinates.

    Returns:
        Events in source order.

    Raises:
        DataValidationError: If the payload is not the expected shape.
    """
    payload: object = json.loads(path.read_text())
    records = _node(payload).get("data")
    if not isinstance(records, list):
        msg = f"{path.name}: expected a top-level 'data' list"
        raise DataValidationError(msg)

    events: list[Event] = []
    for raw in cast("list[object]", records):
        record = _node(raw)
        side = metadata.team_sides.get(_text(_child(record, "team"), "id"))
        if side is None:
            continue
        raw_type = _text(_child(record, "type"), "name").strip().upper()
        start = _child(record, "start")
        end = _child(record, "end")
        start_frame = _int(start, "frame")
        start_time_s = _float(start, "time")

        events.append(
            Event(
                team=side,
                type=EVENT_TYPE_MAP.get(raw_type, EventType.OTHER),
                subtype=_subtype_name(record.get("subtypes")),
                period=_int(record, "period", default=1),
                start_frame=start_frame,
                end_frame=max(_int(end, "frame", default=start_frame), start_frame),
                start_time_s=start_time_s,
                end_time_s=_float(end, "time", default=start_time_s),
                from_player=_optional_text(_child(record, "from"), "id"),
                to_player=_optional_text(_child(record, "to"), "id"),
                start_xy=_xy(start, pitch),
                end_xy=_xy(end, pitch),
                raw_type=raw_type,
            )
        )
    return tuple(events)


# ------------------------------------------------------------------ JSON access
#
# `json.loads` returns Any. These helpers are where that stops: every field the
# event reader touches goes through one of them, so the reader itself deals only
# in `str`, `int`, `float` and `None`. Each preserves the reader's original
# "falsy means absent" convention — a missing key, `null`, `0` and `""` all fall
# back to the default, which is what the previous `x.get(k) or default` chains
# did and what the Metrica exports rely on.


def _node(value: object) -> Mapping[str, object]:
    """Read a value as a JSON object, or an empty one if it is not."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return {}


def _child(node: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Nested object at ``key``, or an empty one when absent or null."""
    return _node(node.get(key))


def _text(node: Mapping[str, object], key: str, default: str = "") -> str:
    """String at ``key``, or ``default`` when absent, null or empty."""
    value = node.get(key)
    if not value:
        return default
    return value if isinstance(value, str) else str(value)


def _optional_text(node: Mapping[str, object], key: str) -> str | None:
    """String at ``key``, or ``None`` when absent, null or empty."""
    value = node.get(key)
    if not value:
        return None
    return value if isinstance(value, str) else str(value)


def _number(node: Mapping[str, object], key: str) -> int | float | str | None:
    """Raw numeric-ish value at ``key``, rejecting types that cannot convert."""
    value = node.get(key)
    if not value:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        msg = f"expected a number for {key!r}, got {type(value).__name__}"
        raise DataValidationError(msg)
    return value


def _int(node: Mapping[str, object], key: str, default: int = 0) -> int:
    """Integer at ``key``, or ``default`` when absent, null or zero."""
    value = _number(node, key)
    return default if value is None else int(value)


def _float(node: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Float at ``key``, or ``default`` when absent, null or zero."""
    value = _number(node, key)
    return default if value is None else float(value)


def _subtype_name(subtypes: object) -> str | None:
    """Flatten the subtype field, which may be an object, a list or null."""
    if isinstance(subtypes, Mapping):
        return _optional_text(cast("Mapping[str, object]", subtypes), "name")
    if isinstance(subtypes, list):
        names = [
            name
            for entry in cast("list[object]", subtypes)
            if (name := _optional_text(_node(entry), "name")) is not None
        ]
        return "-".join(names) if names else None
    return None


def _coordinate(node: Mapping[str, object], key: str) -> float | None:
    """One coordinate component, or ``None`` when absent.

    Unlike :func:`_float` this treats only ``None``/absence as missing: ``0`` is
    a real position on the pitch, not a blank field.
    """
    value = node.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        msg = f"expected a number for coordinate {key!r}, got {type(value).__name__}"
        raise DataValidationError(msg)
    return float(value)


def _xy(node: Mapping[str, object], pitch: Pitch) -> tuple[float, float] | None:
    """Convert an event coordinate to canonical metres, or ``None``."""
    x, y = _coordinate(node, "x"), _coordinate(node, "y")
    if x is None or y is None:
        return None
    canonical = pitch.to_canonical(np.array([x, y]))
    return (float(canonical[0]), float(canonical[1]))


def read_match(
    tracking_path: Path,
    metadata_path: Path,
    events_path: Path,
    pitch: Pitch = DEFAULT_PITCH,
) -> tuple[MatchTracking, tuple[Event, ...], EptsMetadata]:
    """Read a complete EPTS match.

    Args:
        tracking_path: Tracking file.
        metadata_path: Metadata XML.
        events_path: Events JSON.
        pitch: Pitch dimensions.

    Returns:
        Tracking, events and the metadata, which carries the declared
        attacking direction used as tier-1 evidence.
    """
    metadata = read_metadata(metadata_path)
    tracking = read_tracking(tracking_path, metadata, pitch)
    events = read_events(events_path, metadata, pitch)
    return tracking, events, metadata


def declared_directions(metadata: EptsMetadata) -> dict[tuple[int, Team], AttackDirection]:
    """Expand the declared first-half direction to every period.

    Teams change ends at half time, so the second half is the inverse of the
    first. This is a statement of the laws of the game, not an inference, which
    is why it is tier-1 evidence.

    Args:
        metadata: Parsed metadata.

    Returns:
        Direction keyed by period and team; empty when nothing was declared.
    """
    out: dict[tuple[int, Team], AttackDirection] = {}
    for team, direction in metadata.declared_first_half.items():
        for period in metadata.periods:
            out[(period, team)] = direction if period % 2 == 1 else direction.flipped
    return out
