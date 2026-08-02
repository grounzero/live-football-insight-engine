"""Wiring the service together from configuration.

Only startup lives here. The pieces themselves — predictor, match, engine,
player — are built by :mod:`football_insights.serving.loader`, because the
service needs to rebuild them while it is running when someone changes match,
and this module cannot be imported from ``app`` without a cycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from football_insights.replay.player import ReplayPlayer
from football_insights.serving.app import create_app
from football_insights.serving.loader import (
    PREFERRED,
    available_matches,
    build_engine,
    load_match,
    load_predictor,
    load_replay,
)
from football_insights.serving.metrics import Metrics

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
    "load_match",
    "load_predictor",
    "load_replay",
]

LOGGER = logging.getLogger("football_insights.bootstrap")


def create_configured_app(
    settings: Settings,
    match_id: str,
    fault_profile: str = "clean",
    seed: int = 42,
    speed: float = 8.0,
) -> FastAPI:
    """Build the fully wired application.

    Args:
        settings: Resolved configuration.
        match_id: Match to replay.
        fault_profile: Fault profile name.
        seed: Fault-injection seed.
        speed: Replay rate.

    Returns:
        The application, ready to serve.
    """
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
    LOGGER.info(
        "service configured",
        extra={
            "match_id": match_id,
            "fault_profile": fault_profile,
            "seed": seed,
            "speed": speed,
            "model": predictor.metadata.name,
            "is_ml": predictor.metadata.is_ml,
            "threshold": predictor.metadata.threshold,
        },
    )
    return create_app(settings, engine, player, metrics)
