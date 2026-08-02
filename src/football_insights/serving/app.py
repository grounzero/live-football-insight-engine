"""FastAPI service.

Endpoints
---------
``GET  /health``           liveness; always 200 while the process is up.
``GET  /ready``            readiness; false until a predictor is loaded and its
                           feature schema matches this build. Also reports mode,
                           data provenance and predictor identity.
``GET  /capabilities``     which optional surfaces this process exposes.
``GET  /model``            active model metadata, including ``is_ml``.
``GET  /replay/status``    replay position, fault profile and seed.
``GET  /replay/matches``   the match catalogue and which of them are playable.
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
:mod:`football_insights.serving.jobs`. The two mutating replay routes are
withheld the same way when ``service.public_demo`` is set: replay state is
process-wide, so on a public URL one visitor's pause is everyone's pause. In
both cases the routes are not registered at all rather than refused at call
time, so they never appear in the schema and cannot be mistaken for something
temporarily unavailable.

Liveness and readiness are deliberately distinct. A service with no model is
alive and will answer, but is not ready, and says so rather than quietly
returning zeros that a caller might mistake for confident predictions.
Readiness asks whether the service *can* serve, not whether anyone is being
served: the replay task starts on the first subscriber, so requiring a running
replay would mean a deployment health check could never pass before the first
browser arrived.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

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
from football_insights.serving.messages import StreamMessageType
from football_insights.serving.metrics import Metrics
from football_insights.serving.state import AppState, get_state
from football_insights.serving.switching import mount_pipeline_jobs, swap_match
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from football_insights.replay.player import ReplayPlayer
    from football_insights.serving.engine import InsightEngine

LOGGER = logging.getLogger("football_insights.serving")

#: Starlette renamed this constant; resolve it once so the service works on
#: either version without emitting a deprecation warning.
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
#: Renamed in the same Starlette release, and resolved the same way.
HTTP_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)

#: Largest request body accepted. A valid `/predict` window is 50 timesteps of
#: 39 features, roughly 40 kB as JSON, so this is generous by a factor of fifty
#: and still small enough that a body cannot be used to exhaust memory.
MAX_REQUEST_BODY_BYTES: Final = 2 * 1024 * 1024

#: Hard cap on window rows, checked by pydantic before the list is materialised
#: as an array. The exact sequence length a model requires is enforced further
#: down by `validate_batch`; this only stops an absurd allocation.
MAX_WINDOW_TIMESTEPS: Final = 4096

#: Metric label used for requests that matched no route. Without it every
#: scanner probing `/.env` or `/wp-login.php` would add a permanent time series
#: to the registry, which never forgets a label set.
UNMATCHED_ROUTE: Final = "<unmatched>"


class MaxBodySizeMiddleware:
    """Refuse request bodies larger than a fixed cap, while they are read.

    Pure ASGI rather than a ``BaseHTTPMiddleware`` subclass because the limit has
    to apply to ``receive`` itself. Checking ``Content-Length`` alone is not a
    limit: the header is absent under ``Transfer-Encoding: chunked``, and a
    client is free to declare one length and send another. Counting the bytes as
    they arrive is the only version that cannot be talked out of.

    The header is still consulted, purely as a courtesy — an oversized upload
    that announces itself is rejected before a single chunk is read.
    """

    def __init__(self, app: Any, max_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        """Wrap an ASGI application.

        Args:
            app: The application to wrap.
            max_bytes: Largest body accepted, in bytes.
        """
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Pass the request through, counting body bytes on the way.

        The refusal is written from here and the application is then told the
        client went away, rather than an exception being raised for a handler to
        convert. Two layers make that the only version that works:

        * this middleware must sit outside Starlette's ``ExceptionMiddleware``,
          because that is the only position from which ``receive`` can be wrapped
          at all — so an exception raised here reaches the server-error handler
          and becomes a 500;
        * FastAPI wraps body parsing in ``except Exception`` and answers 400, so
          an exception raised inside ``receive`` never escapes the route anyway.

        Sending 413 and following it with ``http.disconnect`` unwinds the handler
        through a path it already understands, and anything it tries to send
        afterwards is dropped because the answer has gone.
        """
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self._max_bytes:
            await self._refuse(send, declared)
            return

        received = 0
        refused = False

        async def counted() -> Any:
            nonlocal received, refused
            if refused:
                return {"type": "http.disconnect"}
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    refused = True
                    await self._refuse(send, received)
                    return {"type": "http.disconnect"}
            return message

        async def guarded(message: Any) -> None:
            # The 413 is already on the wire; the handler's own 400 or 500 is a
            # consequence of the disconnect and must not follow it.
            if not refused:
                await send(message)

        await self._app(scope, counted, guarded)

    async def _refuse(self, send: Any, size: int) -> None:
        """Answer 413 without touching the application."""
        body = json.dumps({"detail": self._message(size)}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": HTTP_413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _message(self, size: int) -> str:
        """Explain the refusal without echoing anything the client sent."""
        return f"request body of {size} bytes exceeds the {self._max_bytes} byte limit"


def _declared_length(scope: Any) -> int | None:
    """Read ``Content-Length`` from a raw ASGI scope, if it is present and sane."""
    for key, value in scope.get("headers", ()):
        if key == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class PredictRequest(BaseModel):
    """An explicit scoring request."""

    window: list[list[float]] = Field(
        ...,
        max_length=MAX_WINDOW_TIMESTEPS,
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


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Shut the background work down when the process is asked to stop.

    The replay task is started lazily by the first subscriber and would
    otherwise be torn down by process death rather than by cancellation, taking
    a pipeline job's child process with it. A container orchestrator sends
    SIGTERM and waits, so there is a window to stop cleanly and it costs
    nothing to use it.
    """
    yield
    state: AppState = app.state.fi
    await state.stop_replay_task()
    manager = getattr(app.state, "fi_jobs", None)
    if manager is not None:
        await manager.shutdown()


def create_app(
    settings: Settings | None = None,
    engine: InsightEngine | None = None,
    player: ReplayPlayer | None = None,
    metrics: Metrics | None = None,
    *,
    data_source: str = "unknown",
) -> FastAPI:
    """Build the application.

    Args:
        settings: Resolved configuration; defaults are used when omitted.
        engine: Inference engine. ``None`` leaves the service alive but not ready.
        player: Replay source for the live stream.
        metrics: Metric sink; a fresh registry is created when omitted.
        data_source: Where the replay's frames came from, reported by ``/ready``
            so a viewer is never left guessing whether a fixture is real.

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
        data_source=data_source,
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
        lifespan=_lifespan,
    )
    app.state.fi = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.service.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(MaxBodySizeMiddleware)

    app.middleware("http")(_correlate)
    app.add_exception_handler(RequestValidationError, _validation_error)

    for router in (OPS_ROUTER, MODEL_ROUTER, REPLAY_ROUTER, INSIGHT_ROUTER):
        app.include_router(router)
    if not resolved.service.public_demo:
        # Withheld rather than refused: on a public URL these would let any
        # visitor pause, rewind or re-pace the replay for everyone else
        # watching, because there is one replay per process and no session.
        app.include_router(REPLAY_CONTROL_ROUTER)
    else:
        LOGGER.info("public demo mode; replay controls are not registered")
    mount_pipeline_jobs(app, state)
    # Last: the demo is mounted at `/`, which would shadow anything registered
    # after it.
    state.ui_available = _mount_demo(app)
    return app


def _endpoint_label(request: Request) -> str:
    """The route template a request matched, never the URL it asked for.

    Prometheus label values become permanent time series: the registry has no
    eviction, so labelling with the raw path would let anyone probing for
    ``/.env`` or ``/wp-login.php`` grow the process's memory without limit, and
    would do the same in normal use through the UUID in ``/jobs/{job_id}``.
    Unmatched requests — every 404 — collapse into one bucket.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else UNMATCHED_ROUTE


async def _correlate(request: Request, call_next: Any) -> Response:
    """Stamp every response with a correlation id and count it."""
    incoming = request.headers.get("x-correlation-id")
    cid = incoming or new_correlation_id()
    response: Response = await call_next(request)
    response.headers["x-correlation-id"] = cid
    state: AppState = request.app.state.fi
    state.metrics.requests.labels(
        endpoint=_endpoint_label(request),
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


def _mount_demo(app: FastAPI) -> bool:
    """Serve the built demo at ``/`` when it exists.

    The demo is an optional build artifact. A missing ``static`` directory means
    ``npm run build`` has not been run; the API is fully functional without it,
    so this is a soft skip rather than an error on a development machine.

    Resolved relative to the installed package rather than the working
    directory, which is what lets the container build put the page inside the
    wheel and have it found wherever the process is started from.

    Returns:
        Whether the page was mounted. Public-demo readiness depends on it: an
        image whose UI failed to build is not a usable public deployment,
        however healthy its API is.
    """
    static_dir = Path(__file__).parent / "static"
    if not (static_dir / "index.html").is_file():
        LOGGER.info("demo not built; serving API only", extra={"static_dir": str(static_dir)})
        return False
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="demo")
    return True


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
#: The mutating half of the replay surface, separated so a public deployment can
#: decline to register it. Read-only replay endpoints stay on `REPLAY_ROUTER`.
REPLAY_CONTROL_ROUTER = APIRouter(prefix="/replay", tags=["replay"])
INSIGHT_ROUTER = APIRouter(tags=["insight"])

StateDep = Annotated[AppState, Depends(get_state)]


@OPS_ROUTER.get("/health")
async def health(st: StateDep) -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": st.settings.service.service_name}


def _readiness_reason(st: AppState) -> str:
    """Why the service is or is not ready, in one phrase.

    Ordered from most fundamental to least, so the answer names the thing to fix
    first rather than the last check that happened to fail.

    The last two conditions apply only to a public demo, and deliberately so.
    Elsewhere a scoring service with no replay and no page is a legitimate
    deployment — ``/predict`` works perfectly well without either. On a public
    URL the replay and the page *are* the product, so an image that reached
    here without them is not something to route visitors to.
    """
    engine = st.engine
    if engine is None:
        return "no engine configured"
    if engine.predictor is None:
        return "no predictor loaded"
    if not engine.is_ready:
        return "feature schema mismatch"
    if st.settings.service.public_demo:
        if st.player is None:
            return "no replay configured"
        if not st.ui_available:
            return "demo page not built"
    return "ok"


def _replay_state(st: AppState) -> str:
    """What the replay is doing, for reporting only.

    Deliberately not part of readiness. The replay task starts on the first
    subscriber, so a service that has never been watched is idle — and a
    deployment health check that ran before the first browser connected would
    never pass if idle counted as unready, leaving the platform to restart a
    perfectly healthy container forever.
    """
    if st.player is None:
        return "unavailable"
    # The task, not the player's own flag. A public replay that ended because
    # nobody was watching leaves the player mid-match and still marked running,
    # so asking the player reports a replay that is not being driven by anything.
    if not st.replay_running:
        return "idle"
    return "paused" if st.player.status().paused else "running"


@OPS_ROUTER.get("/ready")
async def ready(response: Response, st: StateDep) -> JsonDict:
    """Readiness probe.

    Reports ``false`` when no predictor is loaded or its feature schema
    disagrees with this build, rather than serving predictions that cannot
    be trusted.

    Also reports what is being served: which mode, where the frames come from
    and which predictor is answering. That is the same information the page
    shows a viewer, and it comes from here so the two cannot disagree. Nothing
    in the payload is a filesystem path, a credential or a configuration
    fingerprint — it is served unauthenticated on a public URL.
    """
    reason = _readiness_reason(st)
    is_ready = reason == "ok"
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    st.metrics.ready.set(1 if is_ready else 0)

    predictor = st.engine.predictor if st.engine is not None else None
    return {
        "ready": is_ready,
        "reason": reason,
        "mode": "public_demo" if st.settings.service.public_demo else "local",
        "data_source": st.data_source,
        "predictor": None
        if predictor is None
        else {
            "name": predictor.metadata.name,
            "kind": predictor.metadata.kind,
            "is_ml": predictor.metadata.is_ml,
        },
        "ui": st.ui_available,
        "replay": _replay_state(st),
    }


@OPS_ROUTER.get("/capabilities")
async def capabilities(st: StateDep) -> JsonDict:
    """Which optional surfaces this process exposes.

    The demo asks rather than guesses. Pipeline controls are off unless a
    deployment turns them on, and a page that rendered the panel on the
    assumption they were there would offer buttons that 404. Replay controls
    are the same story in reverse: they are on everywhere except a public
    deployment, where the transport is hidden rather than shown broken.
    """
    service = st.settings.service
    return {
        "pipeline_controls": service.enable_pipeline_controls,
        "replay_controls": not service.public_demo,
        "public_demo": service.public_demo,
    }


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
        "matches": list(
            available_matches(
                st.settings.paths.raw_dir, public_demo=st.settings.service.public_demo
            )
        ),
    }


@REPLAY_CONTROL_ROUTER.post("/match")
async def replay_match(command: MatchCommand, st: StateDep) -> JsonDict:
    """Replay a different match, rebuilding the engine in place.

    Roughly two seconds of blocking work, almost all of it parsing tracking
    files, so it runs in a worker thread; holding the event loop for that long
    would stall every other request and every open stream.
    """
    player = _require_player(st)
    catalogue = available_matches(
        st.settings.paths.raw_dir, public_demo=st.settings.service.public_demo
    )
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


@REPLAY_CONTROL_ROUTER.post("/control")
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
                message = await queue.get()
                yield {"event": "update", "data": message.data}
                if message.type is StreamMessageType.END:
                    return
        except asyncio.CancelledError:  # pragma: no cover - client disconnect
            raise
        finally:
            # Sync on purpose. This runs while the generator is being closed,
            # where awaiting is not reliably permitted; the replay loop notices
            # the empty subscriber set on its next frame and ends itself.
            st.unsubscribe(queue)

    return EventSourceResponse(publisher())
