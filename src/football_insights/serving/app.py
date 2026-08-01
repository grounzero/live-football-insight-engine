"""FastAPI service.

Endpoints
---------
``GET  /health``           liveness; always 200 while the process is up.
``GET  /ready``            readiness; false until a predictor is loaded and its
                           feature schema matches this build.
``GET  /model``            active model metadata, including ``is_ml``.
``GET  /replay/status``    replay position, fault profile and seed.
``POST /predict``          score an explicit window.
``GET  /insights/stream``  server-sent events: frames, predictions, insights.
``GET  /metrics``          Prometheus exposition.

Liveness and readiness are deliberately distinct. A service with no model is
alive and will answer, but is not ready, and says so rather than quietly
returning zeros that a caller might mistake for confident predictions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from football_insights.config import Settings
from football_insights.features.spec import DEFAULT_FEATURE_SPEC
from football_insights.insight.types import Insight
from football_insights.serving.logging import configure_logging, new_correlation_id
from football_insights.serving.metrics import Metrics
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from football_insights.replay.player import ReplayPlayer
    from football_insights.serving.engine import InsightEngine

LOGGER = logging.getLogger("football_insights.serving")

#: Starlette renamed this constant; resolve it once so the service works on
#: either version without emitting a deprecation warning.
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)

#: Frames are sent to the browser at this rate regardless of tracking rate;
#: 25 Hz of JSON per client is wasteful and invisible to the eye.
FRAME_PUBLISH_HZ = 12.5


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
    """Pause/resume and pacing control for the replay."""

    paused: bool | None = None
    speed: float | None = Field(default=None, ge=0.0, le=200.0)


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


@dataclass
class AppState:
    """Objects shared across requests."""

    settings: Settings
    metrics: Metrics
    engine: InsightEngine | None = None
    player: ReplayPlayer | None = None
    recent_insights: list[Insight] = field(default_factory=list)
    _subscribers: set[asyncio.Queue[str]] = field(default_factory=set)
    _task: asyncio.Task[None] | None = None

    def subscribe(self) -> asyncio.Queue[str]:
        """Register a new SSE subscriber."""
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        """Remove an SSE subscriber."""
        self._subscribers.discard(queue)

    def publish(self, payload: str, *, critical: bool = False) -> None:
        """Fan out a message, dropping it for any subscriber that has fallen behind.

        A slow browser tab must never stall the replay loop, so a full queue
        loses the message rather than applying back pressure. Control messages
        pass ``critical=True``: losing an end marker would leave that client
        waiting forever, so one slot is made for it by discarding the oldest
        pending frame.
        """
        for queue in list(self._subscribers):
            if queue.full():
                if not critical:
                    continue
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(payload)


def get_state(request: Request) -> AppState:
    """FastAPI dependency returning the shared application state."""
    return request.app.state.fi  # type: ignore[no-any-return]


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

    @app.middleware("http")
    async def _correlate(request: Request, call_next: Any) -> Response:
        incoming = request.headers.get("x-correlation-id")
        cid = incoming or new_correlation_id()
        response: Response = await call_next(request)
        response.headers["x-correlation-id"] = cid
        state.metrics.requests.labels(
            endpoint=request.url.path,
            method=request.method,
            status=str(response.status_code),
        ).inc()
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Return a safe 422 instead of echoing the rejected payload.

        FastAPI's default handler includes the offending input in the response.
        That breaks outright on values JSON cannot represent — a window
        containing ``Infinity`` makes serialising the *error* fail — and echoing
        arbitrary client input back is not something a public endpoint should do
        anyway. Only the location and message are returned.
        """
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

    _register_routes(app, state)
    _mount_demo(app)
    return app


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


def _register_routes(app: FastAPI, state: AppState) -> None:
    """Attach the route handlers."""

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "service": state.settings.service.service_name}

    @app.get("/ready", tags=["ops"])
    async def ready(response: Response, st: Annotated[AppState, Depends(get_state)]) -> JsonDict:
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

    @app.get("/model", tags=["model"])
    async def model(st: Annotated[AppState, Depends(get_state)]) -> JsonDict:
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
        return payload

    @app.get("/replay/status", tags=["replay"])
    async def replay_status(st: Annotated[AppState, Depends(get_state)]) -> JsonDict:
        """Replay position, fault profile and seed."""
        if st.player is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no replay configured"
            )
        return st.player.status().to_dict()

    @app.post("/replay/control", tags=["replay"])
    async def replay_control(
        command: ReplayCommand, st: Annotated[AppState, Depends(get_state)]
    ) -> JsonDict:
        """Pause, resume or re-pace the replay."""
        if st.player is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no replay configured"
            )
        if command.paused is not None:
            st.player.set_paused(command.paused)
        if command.speed is not None:
            st.player.set_speed(command.speed)
        return st.player.status().to_dict()

    @app.post("/predict", response_model=PredictResponse, tags=["model"])
    async def predict(
        request: PredictRequest, st: Annotated[AppState, Depends(get_state)]
    ) -> PredictResponse:
        """Score an explicit observation window."""
        import time

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

    @app.get("/insights", tags=["insight"])
    async def recent(st: Annotated[AppState, Depends(get_state)]) -> JsonDict:
        """Insights emitted so far in this replay."""
        return {"insights": [i.to_dict() for i in st.recent_insights[-20:]]}

    @app.get("/metrics", tags=["ops"])
    async def metrics_endpoint(st: Annotated[AppState, Depends(get_state)]) -> Response:
        """Prometheus exposition."""
        payload, content_type = st.metrics.render()
        return Response(content=payload, media_type=content_type)

    @app.get("/insights/stream", tags=["insight"])
    async def stream(st: Annotated[AppState, Depends(get_state)]) -> EventSourceResponse:
        """Server-sent events carrying frames, predictions and insights."""
        if st.player is None or st.engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no replay configured"
            )
        queue = st.subscribe()
        _ensure_replay_task(st)

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


def _ensure_replay_task(state: AppState) -> None:
    """Start the replay loop once, on first subscriber."""
    if state._task is not None and not state._task.done():
        return
    state._task = asyncio.create_task(_run_replay(state))


async def _run_replay(state: AppState) -> None:
    """Drive the replay through the engine and publish results."""
    player = state.player
    engine = state.engine
    if player is None or engine is None:
        return
    cid = new_correlation_id()
    LOGGER.info(
        "replay started",
        extra={
            "match_id": player.status().match_id,
            "fault_profile": player.status().profile,
            "seed": player.status().seed,
            "speed": player.status().speed,
            "correlation_id": cid,
        },
    )
    publish_every = max(1, round(25.0 / FRAME_PUBLISH_HZ))
    counter = 0
    try:
        async for emitted in player.stream(loop=state.settings.replay.loop):
            result = engine.process(emitted.frame)
            counter += 1

            if result.outcome is not None and result.outcome.insight is not None:
                insight = result.outcome.insight
                state.recent_insights.append(insight)
                state.publish(json.dumps({"type": "insight", "payload": insight.to_dict()}))
                LOGGER.info(
                    "insight emitted",
                    extra={
                        "kind": insight.kind.value,
                        "probability": round(insight.probability, 3),
                        "match_time_s": round(insight.match_time_s, 1),
                        "is_ml": insight.is_ml,
                    },
                )

            if counter % publish_every:
                continue
            frame = emitted.frame
            probability = (
                result.prediction.probability
                if result.prediction and result.prediction.window_valid
                else None
            )
            state.publish(
                json.dumps(
                    {
                        "type": "frame",
                        "payload": {
                            "period": frame.period,
                            "match_time_s": round(frame.time_s, 2),
                            "home": _round_positions(frame.home_xy),
                            "away": _round_positions(frame.away_xy),
                            "ball": _round_positions(frame.ball_xy[None, :])[0],
                            "probability": None if probability is None else round(probability, 4),
                            "window_valid": bool(
                                result.prediction.window_valid if result.prediction else False
                            ),
                        },
                    }
                )
            )
    except asyncio.CancelledError:  # pragma: no cover - shutdown
        raise
    finally:
        state.publish(json.dumps({"type": "end", "payload": {"frames": counter}}), critical=True)
        LOGGER.info("replay finished", extra={"frames": counter})


def _round_positions(xy: np.ndarray) -> list[list[float] | None]:
    """Round coordinates for transport, replacing absent players with ``None``."""
    out: list[list[float] | None] = []
    for row in np.atleast_2d(xy):
        if not np.all(np.isfinite(row)):
            out.append(None)
        else:
            out.append([round(float(row[0]), 2), round(float(row[1]), 2)])
    return out
