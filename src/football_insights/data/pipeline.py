"""Preparation pipeline: raw files to a modelling dataset.

Runs acquisition output through parsing, validation, orientation, causal
possession, feature computation and labelling, and writes a deterministic
processed dataset plus the reports that explain it.

One design point is worth stating because it is easy to get wrong and
impossible to notice afterwards. At serving time a whole observation window is
computed with the team that is in possession **at the prediction instant** —
the rolling buffer has one attacking team, not one per frame. So features are
computed here per ``(period, attacking team)`` pair over the whole period, and
a window is sliced from the matrix belonging to the team in possession at its
final frame. Computing each frame with whichever team happened to hold the ball
at that frame would be a train/serve skew that no test of the feature functions
themselves would catch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from football_insights.data import metrica_csv, metrica_epts
from football_insights.data.acquire import MATCHES_BY_ID, load_manifest
from football_insights.data.orientation import infer_orientation, write_direction_report
from football_insights.data.validate import validate_events, validate_tracking
from football_insights.domain import Team
from football_insights.features.causal import CausalEventView
from football_insights.features.frame_features import (
    PossessionContext,
    box_entry_history,
    box_entry_mask,
    compute_features,
)
from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.labels.box_entry import LabelledWindows, build_labels
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from football_insights.config import Settings
    from football_insights.domain import Event, MatchTracking, Orientation

#: Lookbacks used for the possession-context features, matching the live engine.
RECENT_PASS_LOOKBACK_S = 10.0
BOX_ENTRY_LOOKBACK_S = 120.0


@dataclass(frozen=True, slots=True)
class PreparedMatch:
    """A match ready for modelling."""

    match_id: str
    #: Features per attacking team: ``(2, n_frames, n_features)``, index 0 home.
    features: np.ndarray
    labels: LabelledWindows
    orientation: Orientation
    validation: JsonDict
    frame_rate: float

    def window(self, sample: int, observation_frames: int, sequence_length: int) -> np.ndarray:
        """Materialise one model input.

        Args:
            sample: Index into the labelled samples.
            observation_frames: Raw frames per window.
            sequence_length: Timesteps the model expects.

        Returns:
            Array ``(sequence_length, n_features)``.
        """
        from football_insights.features.window import subsample_indices

        end = int(self.labels.end_index[sample])
        team = int(self.labels.attacking_team[sample])
        start = end - observation_frames + 1
        raw = self.features[team, start : end + 1]
        return np.asarray(raw[subsample_indices(raw.shape[0], sequence_length)])

    def windows(self, observation_frames: int, sequence_length: int) -> np.ndarray:
        """Materialise every model input for this match.

        Returns:
            Array ``(n_samples, sequence_length, n_features)`` in ``float32``.
        """
        from football_insights.features.window import subsample_indices

        n = len(self.labels)
        out = np.empty((n, sequence_length, self.features.shape[2]), dtype=np.float32)
        picks = subsample_indices(observation_frames, sequence_length)
        for k in range(n):
            end = int(self.labels.end_index[k])
            team = int(self.labels.attacking_team[k])
            start = end - observation_frames + 1
            out[k] = self.features[team, start : end + 1][picks]
        return out


def _possession_codes(view: CausalEventView, frames: np.ndarray) -> np.ndarray:
    """Possessing team per frame as ``0`` home, ``1`` away, ``-1`` unknown."""
    codes = np.full(frames.shape[0], -1, dtype=np.int8)
    for i, frame in enumerate(frames):
        team = view.possession(int(frame)).team
        if team is Team.HOME:
            codes[i] = 0
        elif team is Team.AWAY:
            codes[i] = 1
    return codes


def _possession_context(
    view: CausalEventView,
    tracking: MatchTracking,
    in_box: np.ndarray,
    index: np.ndarray,
) -> PossessionContext:
    """Build causal possession context for a set of frames."""
    n = index.shape[0]
    duration = np.zeros(n)
    count = np.zeros(n)
    dead = np.zeros(n)
    flight = np.zeros(n)
    passes = np.zeros(n)
    lookback = int(RECENT_PASS_LOOKBACK_S * tracking.frame_rate)
    from football_insights.domain import EventType

    for k, i in enumerate(index):
        frame = int(tracking.frame[i])
        state = view.possession(frame)
        duration[k] = state.duration_s
        count[k] = state.event_count
        dead[k] = float(state.is_dead_ball)
        flight[k] = float(state.has_event_in_flight)
        counts = view.recent_type_counts(frame, lookback)
        passes[k] = counts.get(EventType.PASS, 0) + counts.get(EventType.CARRY, 0)

    entries, since = box_entry_history(in_box, tracking.time_s[index], BOX_ENTRY_LOOKBACK_S)
    return PossessionContext(
        duration_s=duration,
        event_count=count,
        is_dead_ball=dead,
        event_in_flight=flight,
        recent_pass_count=passes,
        recent_box_entry_count=entries,
        time_since_last_box_entry=since,
    )


def compute_match_features(
    tracking: MatchTracking,
    events: tuple[Event, ...],
    orientation: Orientation,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> np.ndarray:
    """Compute features for both attacking-team hypotheses.

    Args:
        tracking: Parsed tracking.
        events: Parsed events.
        orientation: Attacking direction per period and team.
        spec: Feature schema.

    Returns:
        Array ``(2, n_frames, n_features)``; index 0 is the home team attacking.
    """
    view = CausalEventView(events, tracking.frame_rate)
    n = tracking.n_frames
    out = np.zeros((2, n, spec.n_features), dtype=np.float32)
    gk = {
        Team.HOME: np.array([p.is_goalkeeper for p in tracking.home_players]),
        Team.AWAY: np.array([p.is_goalkeeper for p in tracking.away_players]),
    }

    for period in np.unique(tracking.period):
        index = np.flatnonzero(tracking.period == period)
        if index.size == 0:
            continue
        for code, team in ((0, Team.HOME), (1, Team.AWAY)):
            sign = orientation.direction(int(period), team).sign
            ball = tracking.ball_xy[index]
            in_box = box_entry_mask(ball, sign)
            context = _possession_context(view, tracking, in_box, index)
            out[code, index] = compute_features(
                attack_xy=tracking.team_xy(team)[index],
                defend_xy=tracking.team_xy(team.opponent)[index],
                ball_xy=ball,
                direction_sign=sign,
                frame_rate=tracking.frame_rate,
                possession=context,
                attack_is_gk=gk[team],
                defend_is_gk=gk[team.opponent],
                spec=spec,
            )
    return out


def prepare_match(
    match_id: str,
    settings: Settings,
    raw_root: Path,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> PreparedMatch:
    """Parse, validate, orient, feature-ise and label one match.

    Args:
        match_id: Which match to prepare.
        settings: Resolved configuration.
        raw_root: Directory holding the downloaded data.
        spec: Feature schema.

    Returns:
        The prepared match.
    """
    files = MATCHES_BY_ID[match_id]
    paths = files.paths(raw_root)
    declared = None

    if files.source_format == "metrica_csv":
        tracking, events = metrica_csv.read_match(paths["home"], paths["away"], paths["events"])
    else:
        tracking, events, metadata = metrica_epts.read_match(
            paths["tracking"], paths["metadata"], paths["events"]
        )
        declared = metrica_epts.declared_directions(metadata)

    view = CausalEventView(events, tracking.frame_rate)
    dead_ball = view.dead_ball_frames(tracking.n_frames)
    report = validate_tracking(tracking, match_id, in_play=~dead_ball)
    events = validate_events(events, tracking, match_id, report)

    orientation, home_players, away_players = infer_orientation(
        tracking,
        events,
        match_id,
        declared=declared,
        overrides=settings.direction_overrides,
        override_reasons=settings.direction_override_reasons,
    )
    # Re-bind players so inferred goalkeepers reach the feature computation.
    tracking = type(tracking)(
        period=tracking.period,
        frame=tracking.frame,
        time_s=tracking.time_s,
        home_xy=tracking.home_xy,
        away_xy=tracking.away_xy,
        ball_xy=tracking.ball_xy,
        home_players=home_players,
        away_players=away_players,
        frame_rate=tracking.frame_rate,
    )

    return prepare_parsed_match(
        match_id,
        tracking,
        events,
        orientation,
        settings,
        validation=report.to_dict(),
        spec=spec,
    )


def prepare_parsed_match(
    match_id: str,
    tracking: MatchTracking,
    events: tuple[Event, ...],
    orientation: Orientation,
    settings: Settings,
    *,
    validation: JsonDict | None = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> PreparedMatch:
    """Feature-ise and label a match that is already parsed and oriented.

    Split out of :func:`prepare_match` so a match that never came from a file —
    the generated fixture the demo model is trained on — reaches the modelling
    stage through exactly the same code, rather than through a second
    implementation that could drift from this one.

    Args:
        match_id: Match identifier.
        tracking: Parsed tracking, with goalkeepers already resolved.
        events: Parsed events.
        orientation: Attacking direction per period and team.
        settings: Resolved configuration.
        validation: Validation report, when the caller ran one.
        spec: Feature schema.

    Returns:
        The prepared match.
    """
    view = CausalEventView(events, tracking.frame_rate)
    dead_ball = view.dead_ball_frames(tracking.n_frames)
    possession = _possession_codes(view, tracking.frame)
    features = compute_match_features(tracking, events, orientation, spec)

    labels = build_labels(
        match_id=match_id,
        ball_xy=tracking.ball_xy,
        times_s=tracking.time_s,
        periods=tracking.period,
        possession_team=possession,
        dead_ball=dead_ball,
        orientation=orientation,
        window=settings.window,
        episode_settings=settings.episode,
        frame_rate=tracking.frame_rate,
    )
    return PreparedMatch(
        match_id=match_id,
        features=features,
        labels=labels,
        orientation=orientation,
        validation=validation if validation is not None else {},
        frame_rate=tracking.frame_rate,
    )


def prepare_dataset(
    settings: Settings,
    match_ids: list[str] | None = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> JsonDict:
    """Prepare every match and write the processed dataset and reports.

    Args:
        settings: Resolved configuration.
        match_ids: Matches to prepare; all in the manifest when omitted.
        spec: Feature schema.

    Returns:
        The preparation report.
    """
    raw_root = settings.paths.raw_dir
    manifest = load_manifest(raw_root)
    available = [m["match_id"] for m in manifest["matches"]]
    selected = match_ids or available

    processed = settings.paths.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    reports_dir = settings.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[JsonDict] = []
    for match_id in selected:
        prepared = prepare_match(match_id, settings, raw_root, spec)
        np.savez_compressed(
            processed / f"{match_id}.npz",
            features=prepared.features,
            end_index=prepared.labels.end_index,
            time_s=prepared.labels.time_s,
            period=prepared.labels.period,
            attacking_team=prepared.labels.attacking_team,
            label=prepared.labels.label,
            episode_id=prepared.labels.episode_id,
            cluster_id=prepared.labels.cluster_id,
            episode_time=np.array([e.entry_time_s for e in prepared.labels.episodes]),
            episode_team=np.array(
                [0 if e.team is Team.HOME else 1 for e in prepared.labels.episodes]
            ),
            frame_rate=np.array([prepared.frame_rate]),
        )
        write_direction_report(prepared.orientation, reports_dir / f"direction_{match_id}.json")
        summaries.append(
            {
                "match_id": match_id,
                "validation": prepared.validation,
                "labels": prepared.labels.summary(),
                "orientation": {
                    f"P{p}-{t.value}": d.value
                    for (p, t), d in sorted(prepared.orientation.directions.items())
                },
            }
        )

    report = {
        "dataset_fingerprint": manifest.get("fingerprint"),
        "config_fingerprint": settings.fingerprint(),
        "feature_schema": spec.describe()["schema_hash"],
        "window": settings.window.model_dump(),
        "episode": settings.episode.model_dump(),
        "matches": summaries,
    }
    (reports_dir / "preparation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    return report


def load_prepared(path: Path) -> dict[str, np.ndarray]:
    """Load a processed match written by :func:`prepare_dataset`."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
