"""Running the pipeline stages as tracked background jobs.

Exposes the five stages the Makefile documents — ``data``, ``prepare``,
``train``, ``evaluate``, ``benchmark`` — over HTTP, so the demo can drive them
without a second terminal.

Three decisions are worth stating, because each avoids a specific failure:

**Nothing shells out.** Every Make target is a thin Typer wrapper over one
library call, so the jobs call those functions directly. An HTTP endpoint that
built a shell command would be a far larger thing to get right, and would gain
nothing.

**Each job runs in its own process, not a thread.** These are CPU-bound NumPy and
Torch workloads. In a thread they would hold the GIL against the replay loop and
the live demo would stutter for the whole run. A process also makes cancellation
real: ``ProcessPoolExecutor`` cannot stop a task once it has started, which is no
use for a job whose first act is to download 180 MB.

**Output goes to a file, not through a pipe.** The child redirects its own stdout
and stderr into ``artifacts/jobs/<id>.log``; the parent tails that file for the
log stream. Nothing has to be marshalled between processes, and the log survives
a restart along with the record beside it.

These routes are registered only when ``service.enable_pipeline_controls`` is
set — see :class:`football_insights.config.ServiceSettings` for why that is off
by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import multiprocessing
import os
import sys
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from football_insights.config import Settings
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from multiprocessing.process import BaseProcess

LOGGER = logging.getLogger("football_insights.jobs")

#: Held-out match for ``train``, matching the Makefile's ``TEST_MATCH`` default
#: so a job and `make train` produce the same artifacts.
DEFAULT_TEST_MATCH: Final = "Sample_Game_2"

#: How often the parent checks whether the worker has exited, and how often a
#: log stream looks for new output. Fast enough to feel live, slow enough that a
#: multi-minute job costs a negligible number of wake-ups.
POLL_S: Final = 0.5

#: Records kept in the listing. Older ones stay on disk; this only bounds what a
#: page has to render.
LISTED_RECORDS: Final = 20


# --------------------------------------------------------------- the stages
#
# Each returns a small JSON summary — the same numbers the CLI prints — rather
# than the full report. The reports themselves are written to
# `artifacts/reports/` by the functions below, exactly as the CLI writes them.


def _run_data(settings: Settings) -> JsonDict:
    """Download the Metrica sample data and write a checksummed manifest."""
    from football_insights.data.acquire import acquire

    manifest = acquire(settings.paths.raw_dir, None, False)
    return {
        "fingerprint": manifest["fingerprint"],
        "matches": [entry["match_id"] for entry in manifest["matches"]],
    }


def _run_prepare(settings: Settings) -> JsonDict:
    """Parse, validate, orient, feature-ise and label every match."""
    from football_insights.data.pipeline import prepare_dataset

    report = prepare_dataset(settings, None)
    return {"matches": [entry["labels"] for entry in report["matches"]]}


def _run_train(settings: Settings) -> JsonDict:
    """Train the reference models and register their artifacts."""
    from football_insights.data.acquire import load_manifest
    from football_insights.models.train import train_reference

    fingerprint = load_manifest(settings.paths.raw_dir).get("fingerprint")
    report = train_reference(settings, DEFAULT_TEST_MATCH, None, str(fingerprint))
    return {
        "held_out": DEFAULT_TEST_MATCH,
        "models": {
            name: {"pr_auc": result["window"]["pr_auc"], "episode": result["episode"]}
            for name, result in report["results"]["models"].items()
        },
    }


def _run_evaluate(settings: Settings) -> JsonDict:
    """Leave-one-match-out cross-validation with bootstrap intervals."""
    from football_insights.data.acquire import load_manifest
    from football_insights.models.train import run_cross_validation, write_cross_validation_report

    fingerprint = load_manifest(settings.paths.raw_dir).get("fingerprint")
    report = run_cross_validation(settings, None, str(fingerprint), True)
    write_cross_validation_report(report, settings.paths.reports_dir / "cross_validation.json")
    return {"aggregate": report["aggregate"]}


def _run_benchmark(settings: Settings) -> JsonDict:
    """Benchmark PyTorch against ONNX Runtime and write the report."""
    from football_insights.models.export_onnx import benchmark, export, write_benchmark
    from football_insights.models.temporal import TemporalPredictor
    from football_insights.models.train import load_matches

    predictor = TemporalPredictor.load(settings.paths.registry_dir / "gru-temporal.pt")
    # Always re-export, for the same reason `football-insights benchmark` does:
    # reusing an existing file benchmarks whatever was exported last, which after
    # a retrain is a different model.
    path = export(predictor, settings.paths.registry_dir / "gru-temporal.onnx")
    windows = load_matches(settings)[0].windows[:2000]
    report = benchmark(predictor, path, windows, iterations=300)
    write_benchmark(report, settings.paths.reports_dir / "benchmark.json")
    return {"latency": report["latency"], "parity": report["parity"]}


@dataclass(frozen=True, slots=True)
class JobSpec:
    """One runnable stage.

    ``effect`` names what the running service must do after a successful run.
    A trained model or a re-prepared dataset changes what the process *should*
    be serving, and a service that kept serving the old one without saying so
    would be quietly wrong.
    """

    name: str
    label: str
    description: str
    run: Callable[[Settings], JsonDict]
    effect: str = ""

    def to_dict(self) -> JsonDict:
        """Serialisable form, for the panel that renders the buttons."""
        return {"name": self.name, "label": self.label, "description": self.description}


JOB_SPECS: Final[tuple[JobSpec, ...]] = (
    JobSpec(
        name="data",
        label="Download data",
        description="Fetch the Metrica sample data, about 180 MB, and checksum it.",
        run=_run_data,
    ),
    JobSpec(
        name="prepare",
        label="Prepare",
        description="Parse, validate, orient, feature-ise and label every match.",
        run=_run_prepare,
        effect="reload_match",
    ),
    JobSpec(
        name="train",
        label="Train",
        description="Train the baselines and the GRU, and register the artifacts.",
        run=_run_train,
        effect="reload_model",
    ),
    JobSpec(
        name="evaluate",
        label="Evaluate",
        description="Leave-one-match-out cross-validation with bootstrap intervals.",
        run=_run_evaluate,
    ),
    JobSpec(
        name="benchmark",
        label="Benchmark",
        description="Compare PyTorch against ONNX Runtime and check parity.",
        run=_run_benchmark,
    ),
)

JOBS_BY_NAME: Final = {spec.name: spec for spec in JOB_SPECS}


# ------------------------------------------------------------- the worker


def run_stage(name: str, settings: Settings, result_path: Path) -> None:
    """Run one stage and record its summary. Raises whatever the stage raises.

    Kept separate from :func:`_worker` so it can be called directly: the wrapper
    replaces this process's stdout and stderr, which is right in a child and
    catastrophic anywhere else.
    """
    result = JOBS_BY_NAME[name].run(settings)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n{name} finished", flush=True)


def _worker(name: str, settings: Settings, log_path: Path, result_path: Path) -> None:
    """Child-process entry point: run one stage, with all output in the log file.

    Both streams are redirected at the file-descriptor level rather than by
    rebinding ``sys.stdout``, because the interesting output does not all come
    from Python — Torch and ONNX Runtime write to fd 1 and 2 directly, and a
    log missing exactly the lines a slow run is judged by would be worse than no
    log at all.
    """
    with log_path.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), sys.stdout.fileno())
        os.dup2(handle.fileno(), sys.stderr.fileno())
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
        try:
            run_stage(name, settings, result_path)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
            raise SystemExit(1) from None


def _spawn_worker(name: str, settings: Settings, log_path: Path, result_path: Path) -> BaseProcess:
    """Start the child process for one stage.

    Spawn rather than fork: this parent is an event loop with open sockets and a
    replay task, and forking it into a multi-minute Torch job inherits state that
    then misbehaves.

    ``daemon`` means the worker cannot itself start processes, which is true of
    every stage today and worth keeping true — the alternative is orphaned
    multi-minute jobs surviving a Ctrl-C of the service.
    """
    process = multiprocessing.get_context("spawn").Process(
        target=_worker,
        args=(name, settings, log_path, result_path),
        daemon=True,
    )
    process.start()
    return process


@dataclass
class JobRecord:
    """What happened to one run, and where to read the rest of it."""

    id: str
    name: str
    state: str
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    result: JsonDict | None = None

    @property
    def terminal(self) -> bool:
        """Whether this run has stopped, however it stopped."""
        return self.state in {"succeeded", "failed", "cancelled"}

    def to_dict(self) -> JsonDict:
        """Serialisable form."""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "result": self.result,
        }


@dataclass
class JobManager:
    """Runs one stage at a time and remembers what happened.

    One at a time is a correctness requirement, not a courtesy: two ``prepare``
    runs would write the same ``.npz`` files concurrently, and two ``train`` runs
    the same registry entries.
    """

    settings: Settings
    #: Called with the finished record. Lets the service react to a new model or
    #: a re-prepared dataset without this module knowing anything about the app.
    on_finished: Callable[[JobRecord], Awaitable[None]] | None = None
    #: How a worker is started. Injectable so the bookkeeping here can be tested
    #: without spawning a real interpreter for every case; production never
    #: passes anything but the default. Resolved through a factory rather than
    #: bound as a default value, so the module-level name stays the one place
    #: this is defined.
    spawn: Callable[[str, Settings, Path, Path], BaseProcess] = field(
        default_factory=lambda: _spawn_worker
    )
    records: dict[str, JobRecord] = field(default_factory=dict[str, JobRecord])
    _process: BaseProcess | None = None
    _current: str | None = None
    _watcher: asyncio.Task[None] | None = None

    @property
    def directory(self) -> Path:
        """Where records and logs are written."""
        return self.settings.paths.artifacts_dir / "jobs"

    def log_path(self, job_id: str) -> Path:
        """Log file for a run. Only ever called with an id from ``records``."""
        return self.directory / f"{job_id}.log"

    def _record_path(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.json"

    def result_path(self, job_id: str) -> Path:
        """Where a run's JSON summary is written. Only called with a known id."""
        return self.directory / f"{job_id}.result.json"

    @property
    def running(self) -> JobRecord | None:
        """The run in flight, if there is one."""
        return None if self._current is None else self.records.get(self._current)

    def load(self) -> None:
        """Seed the index from disk, so a restart does not lose the history.

        Anything left marked running belonged to a process that is gone, and is
        recorded as interrupted rather than left looking live forever.
        """
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.json")):
            if path.name.endswith(".result.json"):
                continue
            try:
                data = json.loads(path.read_text())
                record = JobRecord(**data)
            except (OSError, TypeError, ValueError):
                LOGGER.warning("ignoring unreadable job record", extra={"path": str(path)})
                continue
            if not record.terminal:
                record.state = "failed"
                record.error = "the service stopped while this job was running"
            self.records[record.id] = record

    def _persist(self, record: JobRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._record_path(record.id).write_text(json.dumps(record.to_dict(), indent=2) + "\n")

    def start(self, name: str) -> JobRecord:
        """Spawn a worker for one stage.

        Raises:
            KeyError: If the stage is not one this build knows about.
            RuntimeError: If a job is already running.
        """
        spec = JOBS_BY_NAME[name]
        if self.running is not None:
            msg = f"{self._current} is already running"
            raise RuntimeError(msg)

        job_id = f"{name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
        record = JobRecord(id=job_id, name=name, state="running", started_at=_now())
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path(job_id).write_text(f"{spec.label}: {spec.description}\n\n")

        process = self.spawn(name, self.settings, self.log_path(job_id), self.result_path(job_id))

        self._process = process
        self._current = job_id
        self.records[job_id] = record
        self._persist(record)
        self._watcher = asyncio.create_task(self._watch(job_id))
        LOGGER.info("job started", extra={"job": job_id, "pid": process.pid})
        return record

    def cancel(self, job_id: str) -> JobRecord:
        """Stop a running job.

        Raises:
            KeyError: If the id is unknown.
            RuntimeError: If that job is not the one running.
        """
        record = self.records[job_id]
        if self._current != job_id or self._process is None:
            msg = f"{job_id} is not running"
            raise RuntimeError(msg)
        self._process.terminate()
        LOGGER.info("job cancelled", extra={"job": job_id})
        return record

    async def _watch(self, job_id: str) -> None:
        """Wait for the worker to exit and settle the record."""
        process = self._process
        record = self.records[job_id]
        if process is None:  # pragma: no cover - defensive
            return
        try:
            while process.is_alive():
                await asyncio.sleep(POLL_S)
        finally:
            process.join(timeout=POLL_S)
            self._finish(record, process.exitcode)
            self._process = None
            self._current = None
        if self.on_finished is not None:
            await self.on_finished(record)

    def _finish(self, record: JobRecord, exit_code: int | None) -> None:
        """Record how a worker ended, reading its result if it wrote one."""
        record.finished_at = _now()
        record.exit_code = exit_code
        if exit_code == 0:
            record.state = "succeeded"
            path = self.result_path(record.id)
            if path.is_file():
                with contextlib.suppress(OSError, ValueError):
                    record.result = json.loads(path.read_text())
        elif exit_code is not None and exit_code < 0:
            # Negative means killed by a signal, which is how cancellation ends.
            record.state = "cancelled"
            record.error = f"stopped by signal {-exit_code}"
        else:
            record.state = "failed"
            record.error = f"exited with status {exit_code}; see the log"
        self._persist(record)
        LOGGER.info("job finished", extra={"job": record.id, "state": record.state})

    async def shutdown(self) -> None:
        """Stop any running worker. Used by tests and by an orderly shutdown."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher


def _now() -> str:
    """Timestamp in the same shape the rest of the service logs."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- routes


class JobRequest(BaseModel):
    """An empty body, so the stages can grow parameters without a route change."""


def get_jobs(request: Request) -> JobManager:
    """Dependency returning the manager, or 503 when the surface is disabled."""
    manager: JobManager | None = getattr(request.app.state, "fi_jobs", None)
    if manager is None:  # pragma: no cover - the router is not registered then
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="pipeline controls are disabled"
        )
    return manager


JOBS_ROUTER = APIRouter(prefix="/jobs", tags=["jobs"])
JobsDep = Annotated[JobManager, Depends(get_jobs)]


@JOBS_ROUTER.get("")
async def list_jobs(jobs: JobsDep) -> JsonDict:
    """The available stages and the most recent runs, newest first."""
    recent = sorted(jobs.records.values(), key=lambda r: r.started_at, reverse=True)
    return {
        "stages": [spec.to_dict() for spec in JOB_SPECS],
        "running": jobs.running.id if jobs.running else None,
        "jobs": [record.to_dict() for record in recent[:LISTED_RECORDS]],
    }


@JOBS_ROUTER.post("/{name}", status_code=status.HTTP_202_ACCEPTED)
async def start_job(name: str, jobs: JobsDep) -> JsonDict:
    """Start one stage."""
    if name not in JOBS_BY_NAME:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no pipeline stage {name!r}"
        )
    try:
        return jobs.start(name).to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@JOBS_ROUTER.get("/{job_id}")
async def get_job(job_id: str, jobs: JobsDep) -> JsonDict:
    """One run."""
    return _lookup(jobs, job_id).to_dict()


@JOBS_ROUTER.post("/{job_id}/cancel")
async def cancel_job(job_id: str, jobs: JobsDep) -> JsonDict:
    """Stop a running job."""
    record = _lookup(jobs, job_id)
    try:
        return jobs.cancel(record.id).to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@JOBS_ROUTER.get("/{job_id}/log")
async def stream_log(job_id: str, jobs: JobsDep) -> EventSourceResponse:
    """Tail a job's output, closing once the run is over and the file is drained."""
    record = _lookup(jobs, job_id)
    path = jobs.log_path(record.id)

    async def publisher() -> AsyncIterator[dict[str, str]]:
        offset = 0
        while True:
            chunk = ""
            if path.is_file():
                with path.open("r", errors="replace") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                    offset = handle.tell()
            if chunk:
                yield {"event": "update", "data": json.dumps({"type": "log", "text": chunk})}
            if record.terminal:
                # Drained *after* the record went terminal, so nothing written in
                # the worker's last moments is lost by closing on the state alone.
                yield {
                    "event": "update",
                    "data": json.dumps({"type": "done", "record": record.to_dict()}),
                }
                return
            await asyncio.sleep(POLL_S)

    return EventSourceResponse(publisher())


def _lookup(jobs: JobManager, job_id: str) -> JobRecord:
    """Find a record by id.

    Records are looked up in the index and never by building a path from the
    id, so a crafted id cannot address anything outside the jobs directory.
    """
    record = jobs.records.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"no job {job_id!r}")
    return record
