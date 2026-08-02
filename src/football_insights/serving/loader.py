"""Loading the pieces a running service is built from.

Separate from :mod:`football_insights.serving.bootstrap` because the service
itself needs these at *runtime*, not only at startup: switching match rebuilds
the engine and the replay player while the process is up. ``bootstrap`` imports
``create_app`` from ``app``, so ``app`` cannot import ``bootstrap`` back; keeping
the loaders here is what lets both use them.

Loading the best available predictor is deliberately forgiving in one direction
and strict in the other. A missing artifact falls back to the rule-based
predictor, clearly labelled, so the demo runs on a fresh clone before anything
is trained. A *mismatched* artifact does not fall back: a model whose feature
schema disagrees with the running code is refused, readiness goes false, and the
reason is reported. Quietly substituting a different model would be worse than
not starting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from football_insights.data.acquire import AVAILABLE_MATCHES, MATCHES_BY_ID
from football_insights.errors import SchemaVersionError
from football_insights.features.spec import DEFAULT_FEATURE_SPEC
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.replay.player import ReplayPlayer
from football_insights.serving.engine import InsightEngine
from football_insights.serving.metrics import Metrics
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from pathlib import Path

    from football_insights.config import Settings
    from football_insights.domain import Event, MatchTracking, Orientation
    from football_insights.models.base import Predictor

LOGGER = logging.getLogger("football_insights.loader")

#: Preference order when no predictor is named in configuration.
PREFERRED = ("gru-temporal", "baseline-gbdt", "baseline-logistic")


def available_matches(raw_dir: Path) -> tuple[JsonDict, ...]:
    """Every catalogued match, and whether it can actually be replayed.

    Availability is a check against the filesystem rather than a lookup in the
    catalogue. :func:`load_match` reads the raw tracking files, so a match this
    build knows about but has never downloaded is not playable, and offering it
    in a selector would produce a 1.5-second load ending in a stack trace.

    Args:
        raw_dir: Directory the dataset was downloaded into.

    Returns:
        One entry per catalogued match, in catalogue order.
    """
    return tuple(
        {
            "id": match.match_id,
            "source_format": match.source_format,
            "available": all(path.is_file() for path in match.paths(raw_dir).values()),
        }
        for match in AVAILABLE_MATCHES
    )


def load_predictor(settings: Settings) -> Predictor:
    """Load the best available predictor, falling back to the heuristic.

    Args:
        settings: Resolved configuration.

    Returns:
        A predictor. The rule-based fallback is returned when no trained
        artifact is available, with ``is_ml`` false so nothing downstream can
        present it as a model.
    """
    from football_insights.models.baseline import BaselinePredictor
    from football_insights.models.temporal import TemporalPredictor

    registry = settings.paths.registry_dir
    names: list[str] = (
        [settings.model.model_name]
        if settings.model.model_name
        else ([] if settings.model.predictor == "heuristic" else list(PREFERRED))
    )

    for name in names:
        torch_path = registry / f"{name}.pt"
        pickle_path = registry / f"{name}.pkl"
        try:
            if torch_path.is_file():
                predictor = TemporalPredictor.load(torch_path)
            elif pickle_path.is_file():
                predictor = BaselinePredictor.load(pickle_path)  # type: ignore[assignment]
            else:
                continue
        except SchemaVersionError:
            # Do not fall through to another model: a schema mismatch means the
            # feature code has changed under a trained artifact, and the right
            # response is to report it, not to silently serve something else.
            LOGGER.exception("refusing model with mismatched feature schema", extra={"model": name})
            raise
        LOGGER.info("loaded model", extra={"model": name, "is_ml": predictor.metadata.is_ml})
        return predictor

    LOGGER.warning(
        "no trained artifact found; using the rule-based fallback",
        extra={"registry": str(registry), "is_ml": False},
    )
    return HeuristicPredictor(settings.model.threshold)


def load_match(
    settings: Settings, match_id: str
) -> tuple[MatchTracking, tuple[Event, ...], Orientation]:
    """Parse and orient one match for replay.

    Blocking and not cheap — roughly 1.5 seconds per match, almost all of it
    parsing two 32 MB tracking files. Callers on the event loop must run this in
    a worker thread.

    Args:
        settings: Resolved configuration.
        match_id: Which match to load.

    Returns:
        Tracking, events and orientation.
    """
    from football_insights.data import metrica_csv, metrica_epts
    from football_insights.data.orientation import infer_orientation

    files = MATCHES_BY_ID[match_id]
    paths = files.paths(settings.paths.raw_dir)
    declared = None
    if files.source_format == "metrica_csv":
        tracking, events = metrica_csv.read_match(paths["home"], paths["away"], paths["events"])
    else:
        tracking, events, metadata = metrica_epts.read_match(
            paths["tracking"], paths["metadata"], paths["events"]
        )
        declared = metrica_epts.declared_directions(metadata)

    orientation, home_players, away_players = infer_orientation(
        tracking,
        events,
        match_id,
        declared=declared,
        overrides=settings.direction_overrides,
        override_reasons=settings.direction_override_reasons,
    )
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
    return tracking, events, orientation


def build_engine(
    settings: Settings,
    tracking: MatchTracking,
    events: tuple[Event, ...],
    orientation: Orientation,
    predictor: Predictor | None,
    metrics: Metrics,
) -> InsightEngine:
    """Construct the inference engine for a loaded match."""
    threshold = predictor.metadata.threshold if predictor is not None else settings.model.threshold
    tuned = settings.model_copy(deep=True)
    tuned.model.threshold = threshold
    return InsightEngine(
        settings=tuned,
        orientation=orientation,
        events=events,
        frame_rate=tracking.frame_rate,
        predictor=predictor,
        metrics=metrics,
        home_is_gk=np.array([p.is_goalkeeper for p in tracking.home_players]),
        away_is_gk=np.array([p.is_goalkeeper for p in tracking.away_players]),
        spec=DEFAULT_FEATURE_SPEC,
    )


def load_replay(
    settings: Settings,
    match_id: str,
    fault_profile: str,
    seed: int,
    speed: float = 0.0,
) -> tuple[InsightEngine, ReplayPlayer]:
    """Build an engine and a replay player for one match."""
    tracking, events, orientation = load_match(settings, match_id)
    metrics = Metrics()
    predictor = load_predictor(settings)
    engine = build_engine(settings, tracking, events, orientation, predictor, metrics)
    player = ReplayPlayer(
        match_id=match_id,
        tracking=tracking,
        profile=settings.fault_profile(fault_profile),
        seed=seed,
        speed=speed,
    )
    return engine, player


def rebuild_for_match(
    settings: Settings,
    match_id: str,
    predictor: Predictor | None,
    metrics: Metrics,
    fault_profile: str,
    seed: int,
    speed: float,
) -> tuple[InsightEngine, ReplayPlayer]:
    """Build an engine and player for a *different* match on a running service.

    Separate from :func:`load_replay` because the two differ in what they are
    allowed to create. A running service already has a predictor and a metric
    registry, and both must survive the swap: the predictor is match-independent
    so reloading it would cost seconds for nothing, and a second ``Metrics``
    would build a registry that ``GET /metrics`` never reads, leaving every
    counter apparently frozen from the moment someone changed match.

    Args:
        settings: Resolved configuration.
        match_id: Match to switch to.
        predictor: The predictor already in use.
        metrics: The metric registry already being served.
        fault_profile: Fault profile name to carry over.
        seed: Fault-injection seed to carry over.
        speed: Replay rate to carry over.

    Returns:
        The new engine and player. Neither is installed; the caller owns that.
    """
    tracking, events, orientation = load_match(settings, match_id)
    engine = build_engine(settings, tracking, events, orientation, predictor, metrics)
    player = ReplayPlayer(
        match_id=match_id,
        tracking=tracking,
        profile=settings.fault_profile(fault_profile),
        seed=seed,
        speed=speed,
    )
    return engine, player
