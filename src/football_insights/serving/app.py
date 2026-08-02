"""FastAPI service.

Endpoints
---------
``GET  /health``           liveness; always 200 while the process is up.
``GET  /ready``            readiness; false until a predictor is loaded and its
                           feature schema matches this build.
``GET  /capabilities``     which optional surfaces this process exposes.
``GET  /model``            active model metadata, including ``is_ml``.
``GET  /replay/status``    replay position, fault profile and seed.
``GET  /replay/matches``   the match catalogue and which of them are on disk.
``POST /predict``          score an explicit window.
``POST /replay/control``   pause, resume, re-pace or restart the replay.
``POST /replay/match``     replay a different match, in place.
``GET  /insights/stream``  server-sent events. Every message is event name
                           ``update``; the JSON ``type`` discriminates between
                           ``frame``, ``insight``, ``suppression`` (an exact
                           per-reason rollup), ``restart``, ``match`` and
                           ``end``.
``GET  /metrics``          Prometheus exposition.

The pipeline job routes under ``/jobs`` are registered only when
``service.enable_pipeline_controls`` is set; see
:mod:`football_insights.serving.jobs`.

Liveness and readiness are deliberately distinct. A service with no model is
alive and will answer, but is not ready, and says so rather than quietly
returning zeros that a caller might mistake for confident predictions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from football_insights.config import Settings
from football_insights.features.spec import DEFAULT_FEATURE_SPEC
from football_insights.serving.loader import available_matches
from football_insights.serving.logging import configure_logging, new_correlation_id
from football_insights.serving.metrics import Metrics
from football_insights.serving.state import AppState, get_state
from football_insights.serving.switching import mount_pipeline_jobs, swap_match
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from football_insights.replay.player import ReplayPlayer
    from football_insights.serving.engine import InsightEngine

LOGGER = logging.getLogger("football_insights.serving")

#: Starlette renamed this constant; resolve it once so the service works on
#: either version without emitting a deprecation warning.
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


class PredictRequest(BaseModel):
    """An explicit scoring request."""

    window: list[list[float]] = Field(
        ...,
        description="Observation window shaped (sequence_length, n_features).",
    )

    @field_validator("window")
    @classmethod
    def _rectangular_and_finite(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            msg = "window must contain at least one timestep"
            raise ValueError(msg)
        width = len(value[0])
        if width == 0:
            msg = "each timestep must contain at least one feature"
            raise ValueError(msg)
        if any(len(row) != width for row in value):
            msg = "all timesteps must have the same number of features"
            raise ValueError(msg)
        arr = np.asarray(value, dtype=np.float64)
        if not np.isfinite(arr).all():
            msg = "window contains non-finite values"
            raise ValueError(msg)
        return value


class ReplayCommand(BaseModel):
    """Pause/resume, pacing and rewind control for the replay."""

    paused: bool | None = None
    speed: float | None = Field(default=None, ge=0.0, le=200.0)
    #: Rewind to the first frame and rebuild all rolling state. A plain bool
    #: rather than the tri-state the others use: for a command, "not requested"
    #: and "requested false" mean the same thing, where for a setting they do
    #: not.
    restart: bool = False


class MatchCommand(BaseModel):
    """Request to replay a different match.

    Deliberately not a field on :class:`ReplayCommand`. Switching match is slow
    and destructive where the others are instant and idempotent, it needs its
    own not-found and already-in-progress answers, and a single command carrying
    both a match and a speed would be ambiguous about which player the speed was
    meant for.
    """

    match: str


class PredictResponse(BaseModel):
    """Result of an explicit scoring request."""

    probability: float
    threshold: float
    above_threshold: bool
    model_name: str
    model_version: str
    is_ml: bool
    feature_schema: str
    inference_ms: float


def create_app(
    settings: Settings | None = None,
    engine: InsightEngine | None = None,
    player: ReplayPlayer | None = None,
    metrics: Metrics | None = None,
) -> FastAPI:
    """Build the application.

    Args:
        settings: Resolved configuration; defaults are used when omitted.
        engine: Inference engine. ``None`` leaves the service alive but not ready.
        player: Replay source for the live stream.
        metrics: Metric sink; a fresh registry is created when omitted.

    Returns:
        The configured application.
    """
    resolved = settings or Settings()
    configure_logging(resolved.service.log_level)
    state = AppState(
        settings=resolved,
        metrics=metrics or Metrics(),
        engine=engine,
        player=player,
    )
    if engine is not None and engine.is_ready:
        meta = engine.predictor.metadata if engine.predictor else None
        state.metrics.ready.set(1)
        if meta is not None:
            state.metrics.set_model(
                meta.name, meta.version, meta.kind, meta.is_ml, meta.feature_schema_hash
            )

    app = FastAPI(
        title="Live Football Insight Engine",
        version="0.1.0",
        description=(
            "Predicts imminent penalty-area entries from live tracking data and "
            "converts them into qualified, viewer-facing insights."
        ),
    )
    app.state.fi = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.service.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.middleware("http")(_correlate)
    app.add_exception_handler(RequestValidationError, _validation_error)

    for router in (OPS_ROUTER, MODEL_ROUTER, REPLAY_ROUTER, INSIGHT_ROUTER):
        app.include_router(router)
    mount_pipeline_jobs(app, state)
    # Last: the demo is mounted at `/`, which would shadow anything registered
    # after it.
    _mount_demo(app)
    return app


async def _correlate(request: Request, call_next: Any) -> Response:
    """Stamp every response with a correlation id and count it."""
    incoming = request.headers.get("x-correlation-id")
    cid = incoming or new_correlation_id()
    response: Response = await call_next(request)
    response.headers["x-correlation-id"] = cid
    state: AppState = request.app.state.fi
    state.metrics.requests.labels(
        endpoint=request.url.path,
        method=request.method,
        status=str(response.status_code),
    ).inc()
    return response


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    """Return a safe 422 instead of echoing the rejected payload.

    FastAPI's default handler includes the offending input in the response.
    That breaks outright on values JSON cannot represent — a window
    containing ``Infinity`` makes serialising the *error* fail — and echoing
    arbitrary client input back is not something a public endpoint should do
    anyway. Only the location and message are returned.

    Starlette types its exception handlers against the base ``Exception``, so the
    concrete type is narrowed here rather than in the signature.
    """
    if not isinstance(exc, RequestValidationError):  # pragma: no cover - defensive
        raise exc
    return JSONResponse(
        status_code=HTTP_422,
        content={
            "detail": [
                {
                    "loc": [str(part) for part in error.get("loc", ())],
                    "msg": str(error.get("msg", "invalid value")),
                    "type": str(error.get("type", "value_error")),
                }
                for error in exc.errors()
            ]
        },
    )


def _mount_demo(app: FastAPI) -> None:
    """Serve the built demo at ``/`` when it exists.

    The demo is an optional build artifact. A missing ``static`` directory means
    ``npm run build`` has not been run; the API is fully functional without it,
    so this is a soft skip rather than an error.
    """
    static_dir = Path(__file__).parent / "static"
    if not (static_dir / "index.html").is_file():
        LOGGER.info("demo not built; serving API only", extra={"static_dir": str(static_dir)})
        return
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="demo")


# ---------------------------------------------------------------- routers
#
# Routers are built at module level, one per resource, rather than as closures
# inside a single registration function. That keeps each handler independently
# readable and testable, and means shared state arrives the same way everywhere
# — through the `get_state` dependency — instead of being captured from an
# enclosing scope by some handlers and injected into others.

OPS_ROUTER = APIRouter(tags=["ops"])
MODEL_ROUTER = APIRouter(tags=["model"])
REPLAY_ROUTER = APIRouter(prefix="/replay", tags=["replay"])
INSIGHT_ROUTER = APIRouter(tags=["insight"])

StateDep = Annotated[AppState, Depends(get_state)]


@OPS_ROUTER.get("/health")
async def health(st: StateDep) -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": st.settings.service.service_name}


@OPS_ROUTER.get("/ready")
async def ready(response: Response, st: StateDep) -> JsonDict:
    """Readiness probe.

    Reports ``false`` when no predictor is loaded or its feature schema
    disagrees with this build, rather than serving predictions that cannot
    be trusted.
    """
    engine = st.engine
    is_ready = engine is not None and engine.is_ready
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    reason = "ok"
    if engine is None:
        reason = "no engine configured"
    elif engine.predictor is None:
        reason = "no predictor loaded"
    elif not engine.is_ready:
        reason = "feature schema mismatch"
    st.metrics.ready.set(1 if is_ready else 0)
    return {"ready": is_ready, "reason": reason}


@OPS_ROUTER.get("/capabilities")
async def capabilities(st: StateDep) -> JsonDict:
    """Which optional surfaces this process exposes.

    The demo asks rather than guesses. Pipeline controls are off unless a
    deployment turns them on, and a page that rendered the panel on the
    assumption they were there would offer buttons that 404.
    """
    return {"pipeline_controls": st.settings.service.enable_pipeline_controls}


@OPS_ROUTER.get("/metrics")
async def metrics_endpoint(st: StateDep) -> Response:
    """Prometheus exposition."""
    payload, content_type = st.metrics.render()
    return Response(content=payload, media_type=content_type)


@MODEL_ROUTER.get("/model")
async def model(st: StateDep) -> JsonDict:
    """Metadata for the active predictor."""
    if st.engine is None or st.engine.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no predictor loaded",
        )
    meta = st.engine.predictor.metadata
    payload = meta.to_dict()
    payload["running_feature_schema"] = DEFAULT_FEATURE_SPEC.schema_hash
    payload["schema_matches"] = meta.feature_schema_hash == DEFAULT_FEATURE_SPEC.schema_hash
    payload["decision_threshold"] = st.settings.model.threshold
    # The demo names the horizon in its own copy. Exposing it keeps that
    # sentence true if the window is ever retuned, where a hardcoded number in
    # the frontend would quietly start lying.
    payload["horizon_s"] = st.settings.window.horizon_s
    return payload


@MODEL_ROUTER.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, st: StateDep) -> PredictResponse:
    """Score an explicit observation window."""
    if st.engine is None or st.engine.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no predictor loaded"
        )
    predictor = st.engine.predictor
    window = np.asarray(request.window, dtype=np.float32)
    expected = DEFAULT_FEATURE_SPEC.n_features
    if window.shape[1] != expected:
        raise HTTPException(
            status_code=HTTP_422,
            detail=(
                f"expected {expected} features per timestep, got {window.shape[1]}; "
                f"feature schema is {DEFAULT_FEATURE_SPEC.schema_hash}"
            ),
        )
    started = time.perf_counter()
    try:
        probability = float(predictor.predict_proba(window[None, ...])[0])
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_422, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    meta = predictor.metadata
    st.metrics.predictions.labels(model=meta.name, is_ml=str(meta.is_ml).lower()).inc()
    st.metrics.confidence.labels(model=meta.name).observe(probability)
    return PredictResponse(
        probability=probability,
        threshold=st.settings.model.threshold,
        above_threshold=probability >= st.settings.model.threshold,
        model_name=meta.name,
        model_version=meta.version,
        is_ml=meta.is_ml,
        feature_schema=meta.feature_schema_hash,
        inference_ms=elapsed_ms,
    )


def _require_player(st: AppState) -> ReplayPlayer:
    """The replay player, or a 503 when the service was started without one."""
    if st.player is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no replay configured"
        )
    return st.player


@REPLAY_ROUTER.get("/status")
async def replay_status(st: StateDep) -> JsonDict:
    """Replay position, fault profile and seed."""
    return _require_player(st).status().to_dict()


@REPLAY_ROUTER.get("/matches")
async def replay_matches(st: StateDep) -> JsonDict:
    """The match catalogue, and which of them this deployment can actually play."""
    return {
        "current": st.player.status().match_id if st.player is not None else None,
        "matches": list(available_matches(st.settings.paths.raw_dir)),
    }


@REPLAY_ROUTER.post("/match")
async def replay_match(command: MatchCommand, st: StateDep) -> JsonDict:
    """Replay a different match, rebuilding the engine in place.

    Roughly two seconds of blocking work, almost all of it parsing tracking
    files, so it runs in a worker thread; holding the event loop for that long
    would stall every other request and every open stream.
    """
    player = _require_player(st)
    catalogue = available_matches(st.settings.paths.raw_dir)
    if not any(m["id"] == command.match and m["available"] for m in catalogue):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no playable match {command.match!r}; see GET /replay/matches",
        )
    # Checked before acquiring rather than by blocking on the lock: a second
    # request should be told the service is busy, not held open for the two
    # seconds the first one takes. Safe without a further guard because nothing
    # awaits between the test and the acquisition.
    if st.switch.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a match switch is already in progress"
        )
    async with st.switch:
        current = st.engine.predictor if st.engine is not None else None
        return await swap_match(st, command.match, player.status(), current)


@REPLAY_ROUTER.post("/control")
async def replay_control(command: ReplayCommand, st: StateDep) -> JsonDict:
    """Pause, resume, re-pace or restart the replay."""
    player = _require_player(st)
    if command.restart:
        # Rewinding the player here rather than in the loop means `/replay/status`
        # is correct straight away, and a replay whose task has already run to
        # completion is back at frame 0 before a new task is started below.
        #
        # Resuming is not a convenience. A paused loop sits in a sleep and never
        # takes another frame, so it would never reach the point where the
        # engine-side half of the rewind is applied. An explicit `paused` in the
        # same command is honoured below, so a restart-and-stay-paused still
        # ends paused — it just rewinds first.
        player.reset()
        player.set_paused(False)
        st.request_restart()
        st.ensure_replay_task()
    if command.paused is not None:
        player.set_paused(command.paused)
    if command.speed is not None:
        player.set_speed(command.speed)
    return player.status().to_dict()


@INSIGHT_ROUTER.get("/insights")
async def recent(st: StateDep) -> JsonDict:
    """Insights emitted so far in this replay."""
    return {"insights": [i.to_dict() for i in st.recent_insights[-20:]]}


@INSIGHT_ROUTER.get("/insights/stream")
async def stream(st: StateDep) -> EventSourceResponse:
    """Server-sent events carrying frames, predictions and insights."""
    if st.player is None or st.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no replay configured"
        )
    queue = st.subscribe()
    st.ensure_replay_task()

    async def publisher() -> AsyncIterator[dict[str, str]]:
        # The stream is finite when the replay is: on the end marker the
        # generator returns, so the connection closes rather than hanging
        # open on a queue that will never be fed again.
        try:
            while True:
                payload = await queue.get()
                yield {"event": "update", "data": payload}
                if '"type": "end"' in payload:
                    return
        except asyncio.CancelledError:  # pragma: no cover - client disconnect
            raise
        finally:
            st.unsubscribe(queue)

    return EventSourceResponse(publisher())
