"""Rearranging a running service: a different match, or a newly trained model.

Both are the same operation underneath — build a new engine and player, put them
in place of the current ones, and keep every connected client attached while it
happens. The awkward parts are documented where they are handled: the replay
task cannot be reused, and cancelling it must not look like the replay ending.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, status

from football_insights.errors import SchemaVersionError
from football_insights.serving.loader import load_predictor, rebuild_for_match
from football_insights.serving.stream import announce_match
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from football_insights.models.base import Predictor
    from football_insights.replay.player import ReplayStatus
    from football_insights.serving.jobs import JobRecord
    from football_insights.serving.state import AppState

LOGGER = logging.getLogger("football_insights.serving")


def resume_replay(state: AppState) -> None:
    """Restart the replay loop after a swap, but only if anyone is watching.

    The loop is otherwise started by the first subscriber. Starting it here
    unconditionally would replay a whole match into an empty room whenever a
    match was changed from a tab that had since been closed.
    """
    if state.has_subscribers:
        state.ensure_replay_task()


async def swap_match(
    state: AppState, match_id: str, previous: ReplayStatus, predictor: Predictor | None
) -> JsonDict:
    """Replace the engine and player with another match's, keeping clients attached.

    The predictor is passed in rather than reloaded, and the metric registry is
    carried over: a second ``Metrics`` would build a registry that
    ``GET /metrics`` never reads, so every counter would appear frozen from the
    moment someone changed match.

    Subscribers live on the state rather than on the replay task, so a swap does
    not disconnect anyone.

    Raises:
        HTTPException: 500 if the new match cannot be loaded. The old one is put
            back on the air first.
    """
    announce_match(state, match_id, loading=True)
    await state.stop_replay_task()
    try:
        engine, player = await asyncio.to_thread(
            rebuild_for_match,
            state.settings,
            match_id,
            predictor,
            state.metrics,
            previous.profile,
            previous.seed,
            previous.speed,
        )
    except Exception:
        # The old match is still installed and still valid, so put it back on the
        # air rather than leaving a dead service behind a failed switch.
        LOGGER.exception("match switch failed", extra={"match_id": match_id})
        announce_match(state, previous.match_id, loading=False)
        resume_replay(state)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"could not load {match_id}; still playing {previous.match_id}",
        ) from None

    state.engine = engine
    state.player = player
    state.recent_insights.clear()
    resume_replay(state)
    announce_match(state, match_id, loading=False)
    LOGGER.info(
        "match switched",
        extra={"match_id": match_id, "from": previous.match_id, "speed": previous.speed},
    )
    return player.status().to_dict()


def mount_pipeline_jobs(app: FastAPI, state: AppState) -> None:
    """Register the pipeline job surface, if this deployment asked for it.

    Not registered at all when disabled, rather than registered and refusing:
    the endpoints then 404 like any other absent route and never appear in the
    published schema, so nothing advertises a capability this process does not
    have.
    """
    if not state.settings.service.enable_pipeline_controls:
        return
    from football_insights.serving.jobs import JOBS_ROUTER, JobManager

    async def on_finished(record: JobRecord) -> None:
        await apply_job_effect(state, record)

    manager = JobManager(settings=state.settings, on_finished=on_finished)
    manager.load()
    app.state.fi_jobs = manager
    app.include_router(JOBS_ROUTER)
    LOGGER.warning(
        "pipeline controls enabled: /jobs can start long, resource-hungry work "
        "and this service has no authentication",
        extra={"jobs_dir": str(manager.directory)},
    )


async def apply_job_effect(state: AppState, record: JobRecord) -> None:
    """Bring the running service into line with what a finished job changed.

    Without this, a successful train leaves the process serving the predictor it
    loaded at startup, and a re-prepared dataset leaves it replaying a match
    oriented by the previous run — in both cases with nothing on screen saying
    so.

    The match is reloaded rather than the engine patched in place, because the
    engine is built from tracking, events and orientation that the service does
    not keep. Two seconds after a job that took minutes is a fair price for not
    holding a second copy of every match in memory.
    """
    from football_insights.serving.jobs import JOBS_BY_NAME

    if record.state != "succeeded" or state.player is None:
        return
    if not JOBS_BY_NAME[record.name].effect:
        return
    if state.switch.locked():
        LOGGER.warning("skipping post-job reload; a match switch is in progress")
        return

    async with state.switch:
        previous = state.player.status()
        predictor = state.engine.predictor if state.engine is not None else None
        if JOBS_BY_NAME[record.name].effect == "reload_model":
            try:
                predictor = await asyncio.to_thread(load_predictor, state.settings)
            except SchemaVersionError:
                # The refusal is the contract: a model whose feature schema
                # disagrees with this build is not served. Keep the one that
                # works rather than turning a successful train into an outage.
                LOGGER.exception("keeping the current model; the new one has a different schema")
                return
        with contextlib.suppress(HTTPException):
            await swap_match(state, previous.match_id, previous, predictor)
