"""Changing match on a running service, and the gated pipeline job surface.

Both are about a process rearranging itself while it is serving, which is where
the interesting failures live: a replay loop that keeps streaming the match it
was started with, a deliberate cancellation that every browser reads as "the
replay finished", a second job started on top of the first.

Nothing here loads real tracking data. The expensive part is stubbed at
:func:`football_insights.serving.switching.rebuild_for_match`, so what is under
test is the swap, not the parser it delegates to.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from football_insights.config import Settings
from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.replay.player import ReplayPlayer
from football_insights.serving import app as app_module
from football_insights.serving import jobs as jobs_module
from football_insights.serving import switching as switching_module
from football_insights.serving.app import create_app
from football_insights.serving.engine import InsightEngine
from football_insights.serving.loader import available_matches
from football_insights.serving.metrics import Metrics
from football_insights.serving.state import AppState
from football_insights.types import JsonDict
from tests.support import ApiClient

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def match() -> SyntheticMatch:
    """A short synthetic match, shared by every case here."""
    return generate_synthetic_match(seed=11, period_duration_s=20.0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose paths point at a scratch directory, never the real dataset."""
    resolved = Settings()
    resolved.paths.raw_dir = tmp_path / "raw"
    resolved.paths.artifacts_dir = tmp_path / "artifacts"
    return resolved


def _build(
    match: SyntheticMatch, settings: Settings, match_id: str = "synthetic"
) -> tuple[InsightEngine, ReplayPlayer, Metrics]:
    """An engine, a player and the metric registry they share."""
    metrics = Metrics()
    engine = InsightEngine(
        settings=settings,
        orientation=match.orientation,
        events=match.events,
        frame_rate=match.frame_rate,
        predictor=HeuristicPredictor(),
        metrics=metrics,
    )
    player = ReplayPlayer(
        match_id=match_id,
        tracking=match.tracking,
        profile=settings.fault_profile("clean"),
        seed=1,
        speed=0.0,
    )
    return engine, player, metrics


class TestMatchCatalogue:
    """What `available_matches` reports, and what the route does with it."""

    def test_a_catalogued_match_with_no_files_is_not_available(self, tmp_path: Path) -> None:
        entries = available_matches(tmp_path)
        assert [entry["id"] for entry in entries] == [
            "Sample_Game_1",
            "Sample_Game_2",
            "Sample_Game_3",
        ]
        assert not any(entry["available"] for entry in entries)

    def test_availability_is_a_disk_check_not_a_catalogue_lookup(self, tmp_path: Path) -> None:
        # Every file of game 1 present, one file of game 2 missing.
        for name in (
            "Sample_Game_1_RawTrackingData_Home_Team.csv",
            "Sample_Game_1_RawTrackingData_Away_Team.csv",
            "Sample_Game_1_RawEventsData.csv",
        ):
            path = tmp_path / "Sample_Game_1" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        partial = tmp_path / "Sample_Game_2" / "Sample_Game_2_RawEventsData.csv"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text("")

        by_id = {entry["id"]: entry["available"] for entry in available_matches(tmp_path)}
        assert by_id == {"Sample_Game_1": True, "Sample_Game_2": False, "Sample_Game_3": False}

    def test_route_reports_the_current_match(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        engine, player, metrics = _build(match, settings)
        client = ApiClient(TestClient(create_app(settings, engine, player, metrics)))
        body = client.get("/replay/matches").json()
        assert body["current"] == "synthetic"
        assert len(body["matches"]) == 3


class TestMatchSwitchRoute:
    """The route's answers, including the ones it gives when it refuses."""

    @pytest.fixture
    def wired(
        self, match: SyntheticMatch, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[ApiClient, AppState]:
        engine, player, metrics = _build(match, settings)

        def fake_catalogue(_: Path, *, public_demo: bool = False) -> tuple[JsonDict, ...]:
            return (
                {"id": "Sample_Game_1", "source_format": "metrica_csv", "available": True},
                {"id": "Sample_Game_3", "source_format": "metrica_epts", "available": False},
            )

        def fake_rebuild(
            _settings: Settings, match_id: str, *rest: object
        ) -> tuple[InsightEngine, ReplayPlayer]:
            # Positional, matching `rebuild_for_match`: predictor, metrics,
            # profile, seed, speed. Only the last three are asserted on.
            _predictor, _metrics, profile, seed, speed = rest
            new_engine, _, _ = _build(match, settings)
            return new_engine, ReplayPlayer(
                match_id=match_id,
                tracking=match.tracking,
                profile=settings.fault_profile(str(profile)),
                seed=int(seed),  # type: ignore[call-overload]
                speed=float(speed),  # type: ignore[arg-type]
            )

        monkeypatch.setattr(app_module, "available_matches", fake_catalogue)
        monkeypatch.setattr(switching_module, "rebuild_for_match", fake_rebuild)
        application = create_app(settings, engine, player, metrics)
        state: AppState = application.state.fi
        return ApiClient(TestClient(application)), state

    def test_unknown_match_is_404(self, wired: tuple[ApiClient, AppState]) -> None:
        client, _ = wired
        response = client.post("/replay/match", json={"match": "Sample_Game_9"})
        assert response.status_code == 404

    def test_a_catalogued_but_undownloaded_match_is_404(
        self, wired: tuple[ApiClient, AppState]
    ) -> None:
        """Offering it would produce a slow load ending in a stack trace."""
        client, _ = wired
        response = client.post("/replay/match", json={"match": "Sample_Game_3"})
        assert response.status_code == 404

    def test_switching_returns_the_new_status(self, wired: tuple[ApiClient, AppState]) -> None:
        client, _ = wired
        body = client.post("/replay/match", json={"match": "Sample_Game_1"}).json()
        assert body["match_id"] == "Sample_Game_1"
        assert client.get("/replay/status").json()["match_id"] == "Sample_Game_1"

    def test_pacing_and_fault_settings_carry_over(self, wired: tuple[ApiClient, AppState]) -> None:
        client, _ = wired
        client.post("/replay/control", json={"speed": 4.0})
        body = client.post("/replay/match", json={"match": "Sample_Game_1"}).json()
        assert body["speed"] == 4.0
        assert body["seed"] == 1
        assert body["fault_profile"] == "clean"

    def test_a_missing_match_field_is_422(self, wired: tuple[ApiClient, AppState]) -> None:
        client, _ = wired
        assert client.post("/replay/match", json={}).status_code == 422

    def test_the_engine_is_replaced_and_history_cleared(
        self, wired: tuple[ApiClient, AppState]
    ) -> None:
        client, state = wired
        before_engine, before_player, metrics = state.engine, state.player, state.metrics
        state.recent_insights.append(object())  # type: ignore[arg-type]

        client.post("/replay/match", json={"match": "Sample_Game_1"})

        assert state.engine is not before_engine
        assert state.player is not before_player
        assert state.recent_insights == []
        # The registry is carried over, not rebuilt: a second one would leave
        # `/metrics` reading counters that nothing increments.
        assert state.metrics is metrics

    def test_a_failed_load_keeps_the_previous_match(
        self, wired: tuple[ApiClient, AppState], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, state = wired
        before_engine, before_player = state.engine, state.player

        def boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("no such file")

        monkeypatch.setattr(switching_module, "rebuild_for_match", boom)
        response = client.post("/replay/match", json={"match": "Sample_Game_1"})

        assert response.status_code == 500
        assert state.engine is before_engine
        assert state.player is before_player

    def test_a_swap_with_nobody_watching_starts_no_replay(
        self, wired: tuple[ApiClient, AppState]
    ) -> None:
        """Otherwise a switch from a tab that has since closed replays to nobody."""
        client, state = wired
        client.post("/replay/match", json={"match": "Sample_Game_1"})
        assert state.player is not None
        assert state.player.status().frames_emitted == 0
        assert state.player.status().running is False


class TestSwapMechanics:
    """The parts a route test cannot see, driven directly against `AppState`."""

    async def test_stopping_the_task_publishes_no_end_marker(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        """A swap must not look like a finished replay.

        Clients close their stream for good on an end marker, so publishing one
        while cancelling on purpose would disconnect every open tab and present
        a match change as the end of the match.
        """
        engine, player, metrics = _build(match, settings)
        state = AppState(settings=settings, metrics=metrics, engine=engine, player=player)
        queue = state.subscribe()
        state.ensure_replay_task()
        await asyncio.sleep(0.05)
        await state.stop_replay_task()

        published = [json.loads(queue.get_nowait().data) for _ in range(queue.qsize())]
        assert published, "the loop should have published something before being stopped"
        assert not any(message["type"] == "end" for message in published)

    async def test_a_replay_that_finishes_on_its_own_still_ends(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        engine, player, metrics = _build(match, settings)
        state = AppState(settings=settings, metrics=metrics, engine=engine, player=player)
        queue = state.subscribe()
        state.ensure_replay_task()
        for _ in range(200):
            await asyncio.sleep(0.02)
            if not state.has_subscribers:  # pragma: no cover - defensive
                break
            drained = [json.loads(queue.get_nowait().data) for _ in range(queue.qsize())]
            if any(message["type"] == "end" for message in drained):
                return
        pytest.fail("the replay never published an end marker")

    async def test_stopping_a_task_that_already_finished_is_a_no_op(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        """Its end marker was published legitimately and clients have acted on it."""
        engine, player, metrics = _build(match, settings)
        state = AppState(settings=settings, metrics=metrics, engine=engine, player=player)
        queue = state.subscribe()
        state.ensure_replay_task()
        while queue.empty():
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        await state.stop_replay_task()
        await state.stop_replay_task()


class TestPipelineGate:
    """The job surface exists only where a deployment has asked for it."""

    def test_capabilities_reports_the_gate(self, match: SyntheticMatch, settings: Settings) -> None:
        engine, player, metrics = _build(match, settings)
        client = ApiClient(TestClient(create_app(settings, engine, player, metrics)))
        assert client.get("/capabilities").json() == {
            "pipeline_controls": False,
            "replay_controls": True,
            "public_demo": False,
        }

    def test_the_routes_are_absent_when_disabled(self, settings: Settings) -> None:
        application = create_app(settings)
        assert not any(path.startswith("/jobs") for path in application.openapi()["paths"])
        assert ApiClient(TestClient(application)).get("/jobs").status_code == 404

    def test_the_routes_appear_when_enabled(self, settings: Settings) -> None:
        settings.service.enable_pipeline_controls = True
        application = create_app(settings)
        assert "/jobs" in application.openapi()["paths"]
        client = ApiClient(TestClient(application))
        assert client.get("/capabilities").json() == {
            "pipeline_controls": True,
            "replay_controls": True,
            "public_demo": False,
        }
        listing = client.get("/jobs").json()
        assert [stage["name"] for stage in listing["stages"]] == [
            "data",
            "prepare",
            "train",
            "evaluate",
            "benchmark",
        ]
        assert listing["running"] is None


class _FakeProcess:
    """A worker that never runs, so the bookkeeping can be tested in isolation."""

    def __init__(self, exitcode: int | None = None) -> None:
        self.pid = 4242
        self.exitcode = exitcode
        self.terminated = False
        self._alive = exitcode is None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self.terminated = True
        self.exitcode = -15
        self._alive = False

    def finish(self, exitcode: int) -> None:
        """Stand in for the process exiting."""
        self.exitcode = exitcode
        self._alive = False


def _manager(settings: Settings, process: _FakeProcess, **kwargs: Any) -> jobs_module.JobManager:
    """A manager whose worker is the fake above."""
    return jobs_module.JobManager(
        settings=settings,
        spawn=lambda *_args: process,  # type: ignore[arg-type]
        **kwargs,
    )


class TestJobExecution:
    """What the child does, tested without letting it near this process's stdout."""

    def test_a_stage_writes_its_summary(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = jobs_module.JobSpec(
            name="fake", label="Fake", description="", run=lambda _s: {"ok": True}
        )
        monkeypatch.setitem(jobs_module.JOBS_BY_NAME, "fake", spec)
        result = tmp_path / "result.json"
        jobs_module.run_stage("fake", settings, result)
        assert json.loads(result.read_text()) == {"ok": True}

    def test_a_failing_stage_raises_rather_than_writing_a_result(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(_s: Settings) -> JsonDict:
            raise RuntimeError("the dataset is not there")

        spec = jobs_module.JobSpec(name="fake", label="Fake", description="", run=explode)
        monkeypatch.setitem(jobs_module.JOBS_BY_NAME, "fake", spec)
        result = tmp_path / "result.json"
        with pytest.raises(RuntimeError, match="not there"):
            jobs_module.run_stage("fake", settings, result)
        assert not result.exists()


class TestJobManager:
    """Single-flight, record settling and what survives a restart."""

    async def test_one_job_at_a_time(self, settings: Settings) -> None:
        manager = _manager(settings, _FakeProcess())
        record = manager.start("prepare")
        assert record.state == "running"
        with pytest.raises(RuntimeError, match="already running"):
            manager.start("train")
        await manager.shutdown()

    async def test_a_clean_exit_is_recorded_with_its_result(self, settings: Settings) -> None:
        process = _FakeProcess()
        manager = _manager(settings, process)
        record = manager.start("prepare")
        manager.result_path(record.id).write_text(json.dumps({"matches": []}))
        process.finish(0)
        await asyncio.sleep(jobs_module.POLL_S * 3)

        assert record.state == "succeeded"
        assert record.result == {"matches": []}
        assert manager.running is None

    async def test_a_signalled_exit_reads_as_cancelled(self, settings: Settings) -> None:
        """Negative exit codes mean a signal, which is how cancellation ends."""
        process = _FakeProcess()
        manager = _manager(settings, process)
        record = manager.start("prepare")
        manager.cancel(record.id)
        await asyncio.sleep(jobs_module.POLL_S * 3)

        assert process.terminated
        assert record.state == "cancelled"

    async def test_a_nonzero_exit_reads_as_failed(self, settings: Settings) -> None:
        process = _FakeProcess()
        manager = _manager(settings, process)
        record = manager.start("prepare")
        process.finish(1)
        await asyncio.sleep(jobs_module.POLL_S * 3)

        assert record.state == "failed"
        assert record.error is not None

    async def test_cancelling_something_that_is_not_running_is_refused(
        self, settings: Settings
    ) -> None:
        process = _FakeProcess()
        manager = _manager(settings, process)
        record = manager.start("prepare")
        process.finish(0)
        await asyncio.sleep(jobs_module.POLL_S * 3)
        with pytest.raises(RuntimeError, match="not running"):
            manager.cancel(record.id)

    async def test_an_effect_runs_only_after_a_successful_stage(self, settings: Settings) -> None:
        seen: list[str] = []

        async def on_finished(record: jobs_module.JobRecord) -> None:
            seen.append(record.state)

        process = _FakeProcess()
        manager = _manager(settings, process, on_finished=on_finished)
        manager.start("prepare")
        process.finish(0)
        await asyncio.sleep(jobs_module.POLL_S * 3)
        assert seen == ["succeeded"]

    async def test_a_run_interrupted_by_a_restart_is_not_left_looking_live(
        self, settings: Settings
    ) -> None:
        process = _FakeProcess()
        manager = _manager(settings, process)
        record = manager.start("prepare")
        await manager.shutdown()

        revived = _manager(settings, _FakeProcess())
        revived.load()
        assert revived.records[record.id].state == "failed"
        assert revived.running is None


class TestJobRoutes:
    """The HTTP surface, with the worker stubbed out."""

    @pytest.fixture
    def client(self, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[ApiClient]:
        settings.service.enable_pipeline_controls = True

        def spawn(*_args: object) -> _FakeProcess:
            return _FakeProcess()

        monkeypatch.setattr(jobs_module, "_spawn_worker", spawn)
        # Entered as a context manager so the event loop outlives each request.
        # Without it the watcher task is cancelled the moment a response is
        # returned, and every job settles as failed before the next call.
        with TestClient(create_app(settings)) as client:
            yield ApiClient(client)

    def test_an_unknown_stage_is_404(self, client: ApiClient) -> None:
        assert client.post("/jobs/rm-rf").status_code == 404

    def test_an_unknown_job_id_is_404(self, client: ApiClient) -> None:
        """Ids are looked up in the index, never turned into a path."""
        assert client.get("/jobs/../../etc/passwd").status_code == 404

    def test_starting_a_stage_is_accepted_and_then_busy(self, client: ApiClient) -> None:
        first = client.post("/jobs/prepare")
        assert first.status_code == 202
        assert first.json()["state"] == "running"
        second = client.post("/jobs/train")
        assert second.status_code == 409
        assert client.get("/jobs").json()["running"] == first.json()["id"]
