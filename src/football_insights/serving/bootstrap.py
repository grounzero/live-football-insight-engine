"""Wiring the service together from configuration.

Only startup lives here. The pieces themselves — predictor, match, engine,
player — are built by :mod:`football_insights.serving.loader`, because the
service needs to rebuild them while it is running when someone changes match,
and this module cannot be imported from ``app`` without a cycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from football_insights.config import resolve_replay_speed
from football_insights.replay.player import ReplayPlayer
from football_insights.serving.app import create_app
from football_insights.serving.loader import (
    PREFERRED,
    available_matches,
    build_engine,
    default_match_id,
    demo_fixture_cycle,
    load_match,
    load_predictor,
    load_replay,
    resolve_match_source,
)
from football_insights.serving.metrics import Metrics
from football_insights.serving.state import FixtureRotation

if TYPE_CHECKING:
    from fastapi import FastAPI

    from football_insights.config import Settings

#: Re-exported for callers that predate the split; ``cli.py`` imports
#: ``load_replay`` from here.
__all__ = [
    "PREFERRED",
    "available_matches",
    "build_engine",
    "create_configured_app",
    "default_match_id",
    "load_match",
    "load_predictor",
    "load_replay",
]

LOGGER = logging.getLogger("football_insights.bootstrap")


def create_configured_app(
    settings: Settings,
    match_id: str | None = None,
    fault_profile: str = "clean",
    seed: int = 42,
    speed: float | None = None,
) -> FastAPI:
    """Build the fully wired application.

    Args:
        settings: Resolved configuration.
        match_id: Match to replay; resolved from configuration and mode when
            omitted, which in public-demo mode means the generated fixture.
        fault_profile: Fault profile name.
        seed: Fault-injection seed.
        speed: Replay rate. ``None`` resolves it from the settings and the mode,
            so a caller that has no opinion gets the public demo's 4x on a
            public deployment and the local default elsewhere, rather than
            whichever number this signature happened to name.

    Returns:
        The application, ready to serve.
    """
    resolved_speed = resolve_replay_speed(speed, settings)
    resolved_id = match_id or default_match_id(settings)

    # The public rotation is generated once, here, rather than at each
    # changeover: a five-minute fixture takes about a quarter of a second to
    # produce, which is a long time to stall the event loop in the middle of a
    # replay every viewer is watching. Three of them add well under a second to
    # startup, and the health check allows minutes.
    #
    # Only when nobody named a match. An explicit `--match` is a request to play
    # that one, and rotating away from it would ignore the instruction.
    rotation: list[FixtureRotation] = []
    if settings.service.public_demo and match_id is None and not settings.replay.match_id:
        for fixture in demo_fixture_cycle():
            loaded_tracking, loaded_events, loaded_orientation = fixture.load()
            rotation.append(
                FixtureRotation(
                    match_id=fixture.match_id,
                    name=fixture.profile.name,
                    narrative=fixture.profile.narrative,
                    tracking=loaded_tracking,
                    events=loaded_events,
                    orientation=loaded_orientation,
                )
            )

    if rotation:
        # The first of the rotation is what plays first; loading it again
        # through `resolve_match_source` would generate the same fixture twice.
        first = rotation[0]
        resolved_id = first.match_id
        source = resolve_match_source(settings, resolved_id)
        tracking, events, orientation = first.tracking, first.events, first.orientation
    else:
        source = resolve_match_source(settings, resolved_id)
        tracking, events, orientation = source.load()
    metrics = Metrics()
    predictor = load_predictor(settings)
    engine = build_engine(settings, tracking, events, orientation, predictor, metrics)
    player = ReplayPlayer(
        match_id=resolved_id,
        tracking=tracking,
        profile=settings.fault_profile(fault_profile),
        seed=seed,
        speed=resolved_speed,
    )
    LOGGER.info(
        "service configured",
        extra={
            "match_id": resolved_id,
            "data_source": source.data_source,
            "public_demo": settings.service.public_demo,
            "fault_profile": fault_profile,
            "seed": seed,
            "speed": resolved_speed,
            "model": predictor.metadata.name,
            "is_ml": predictor.metadata.is_ml,
            "threshold": predictor.metadata.threshold,
        },
    )
    app = create_app(settings, engine, player, metrics, data_source=source.data_source)
    app.state.fi.fixtures = tuple(rotation)
    return app
