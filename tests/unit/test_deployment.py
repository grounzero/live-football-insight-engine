"""What the published container depends on.

These cover the difference between a service that runs on the machine it was
built on and one that can be handed to a hosting platform: where the port comes
from, whether a match can be produced without a dataset, what readiness means
before anyone is watching, and which endpoints a public visitor can reach.

Everything here runs without Docker. The container itself is exercised by
`scripts/container-smoke.sh`, which starts the built image and drives it over
HTTP; these tests are the layer below that, so a failure points at a function
rather than at a container.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from football_insights.config import Settings, resolve_host, resolve_port
from football_insights.data.synthetic import generate_synthetic_match
from football_insights.errors import ConfigurationError, DataValidationError
from football_insights.features.spec import DEFAULT_FEATURE_SPEC
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.replay.player import ReplayPlayer
from football_insights.serving import app as app_module
from football_insights.serving.app import (
    MAX_REQUEST_BODY_BYTES,
    UNMATCHED_ROUTE,
    create_app,
)
from football_insights.serving.bootstrap import create_configured_app
from football_insights.serving.loader import (
    SYNTHETIC_MATCH_ID,
    MetricaMatchSource,
    SyntheticDemoMatchSource,
    available_matches,
    build_engine,
    default_match_id,
    load_predictor,
    resolve_match_source,
)
from football_insights.serving.metrics import Metrics
from football_insights.serving.state import AppState
from football_insights.serving.stream import apply_restart
from tests.support import ApiClient

if TYPE_CHECKING:
    from collections.abc import Iterator

    from football_insights.data.synthetic import SyntheticMatch
    from football_insights.serving.engine import InsightEngine

#: Somewhere that certainly holds neither a dataset nor a model registry. Both
#: the raw and registry directories point here in the tests below, which is what
#: makes "starts with nothing on disk" a claim rather than an assumption.
NOWHERE = Path("/nonexistent/football-insights/should-not-exist")


def _public_settings(tmp_path: Path) -> Settings:
    """Settings shaped like the container's, with nothing available on disk."""
    settings = Settings()
    settings.service.public_demo = True
    settings.paths.raw_dir = NOWHERE / "raw"
    settings.paths.registry_dir = NOWHERE / "registry"
    settings.paths.artifacts_dir = tmp_path / "artifacts"
    return settings


class TestPortResolution:
    """Where the listen port comes from, and what is refused."""

    def test_the_default_is_the_configured_port(self) -> None:
        assert resolve_port(None, Settings(), env={}) == 8000

    def test_the_platform_variable_is_honoured(self) -> None:
        assert resolve_port(None, Settings(), env={"PORT": "8087"}) == 8087

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        # Platform-injected values arrive from a shell, not from a person.
        assert resolve_port(None, Settings(), env={"PORT": " 8087\n"}) == 8087

    def test_an_explicit_option_beats_the_platform(self) -> None:
        assert resolve_port(9001, Settings(), env={"PORT": "8087"}) == 9001

    def test_the_platform_beats_project_configuration(self) -> None:
        """A baked-in FI_SERVICE__PORT must not outrank the host's routing.

        The platform sends traffic to PORT regardless of what the image was
        built with, so preferring the image's own value produces a container
        that passes every check and answers nothing.
        """
        settings = Settings()
        settings.service.port = 9999
        assert resolve_port(None, settings, env={"PORT": "8087"}) == 8087

    def test_project_configuration_applies_when_the_platform_is_silent(self) -> None:
        settings = Settings()
        settings.service.port = 9999
        assert resolve_port(None, settings, env={}) == 9999

    def test_an_empty_platform_variable_is_treated_as_unset(self) -> None:
        assert resolve_port(None, Settings(), env={"PORT": ""}) == 8000

    @pytest.mark.parametrize("value", ["", "http", "80.5", "8O87", "eighty"])
    def test_a_non_numeric_port_is_refused(self, value: str) -> None:
        if not value:
            pytest.skip("empty is unset, covered above")
        with pytest.raises(ConfigurationError, match="not an integer"):
            resolve_port(None, Settings(), env={"PORT": value})

    @pytest.mark.parametrize("value", [0, -1, 65536, 1_000_000])
    def test_an_out_of_range_port_is_refused(self, value: int) -> None:
        with pytest.raises(ConfigurationError, match="valid port range"):
            resolve_port(None, Settings(), env={"PORT": str(value)})

    @pytest.mark.parametrize("value", [0, -1, 65536])
    def test_an_out_of_range_option_is_refused(self, value: int) -> None:
        # `--port 0` used to be swallowed by an `or`, which silently served the
        # configured port instead of reporting an impossible request.
        with pytest.raises(ConfigurationError, match="valid port range"):
            resolve_port(value, Settings(), env={})

    @pytest.mark.parametrize("value", [1, 80, 8000, 65535])
    def test_the_whole_valid_range_is_accepted(self, value: int) -> None:
        assert resolve_port(None, Settings(), env={"PORT": str(value)}) == value

    def test_host_prefers_the_option_then_configuration(self) -> None:
        settings = Settings()
        settings.service.host = "0.0.0.0"
        assert resolve_host(None, settings) == "0.0.0.0"
        assert resolve_host("127.0.0.1", settings) == "127.0.0.1"


class TestPublicDemoSettings:
    """The flag, and how it is spelled."""

    def test_it_is_off_by_default(self) -> None:
        assert Settings().service.public_demo is False

    def test_the_environment_variable_turns_it_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FI_SERVICE__PUBLIC_DEMO", "1")
        assert Settings().service.public_demo is True

    def test_the_default_match_follows_the_mode(self) -> None:
        local = Settings()
        assert default_match_id(local) == "Sample_Game_2"

        public = Settings()
        public.service.public_demo = True
        assert default_match_id(public) == SYNTHETIC_MATCH_ID

    def test_an_explicit_configured_match_wins_in_either_mode(self) -> None:
        settings = Settings()
        settings.service.public_demo = True
        settings.replay.match_id = "Sample_Game_1"
        assert default_match_id(settings) == "Sample_Game_1"


class TestMatchSourceSelection:
    """Which source a match id resolves to."""

    def test_the_fixture_id_selects_the_generator(self) -> None:
        source = resolve_match_source(Settings(), SYNTHETIC_MATCH_ID)
        assert isinstance(source, SyntheticDemoMatchSource)
        assert source.data_source == "synthetic"

    def test_any_other_id_selects_the_dataset(self) -> None:
        source = resolve_match_source(Settings(), "Sample_Game_2")
        assert isinstance(source, MetricaMatchSource)
        assert source.data_source == "metrica"

    def test_the_fixture_is_available_outside_public_mode_too(self) -> None:
        """Keyed on the id, not the mode.

        A developer with no dataset can still run the service, and an explicit
        request for a real match never silently becomes the fixture.
        """
        settings = Settings()
        assert settings.service.public_demo is False
        assert isinstance(
            resolve_match_source(settings, SYNTHETIC_MATCH_ID), SyntheticDemoMatchSource
        )

    def test_public_mode_lists_only_the_fixture(self) -> None:
        catalogue = available_matches(NOWHERE, public_demo=True)
        assert [m["id"] for m in catalogue] == [SYNTHETIC_MATCH_ID]
        assert catalogue[0]["available"] is True

    def test_local_mode_still_lists_the_dataset(self) -> None:
        catalogue = available_matches(NOWHERE)
        assert [m["id"] for m in catalogue] == [
            "Sample_Game_1",
            "Sample_Game_2",
            "Sample_Game_3",
        ]
        assert all(m["available"] is False for m in catalogue)


class TestSyntheticSource:
    """The fixture the hosted demo replays."""

    def test_it_needs_no_filesystem(self) -> None:
        settings = Settings()
        settings.paths.raw_dir = NOWHERE / "raw"
        tracking, events, orientation = resolve_match_source(settings, SYNTHETIC_MATCH_ID).load()

        assert tracking.n_frames > 0
        assert events
        assert orientation.report["source"] == "synthetic"

    def test_it_is_deterministic_for_a_seed(self) -> None:
        first = SyntheticDemoMatchSource(seed=42).load()
        second = SyntheticDemoMatchSource(seed=42).load()

        assert np.array_equal(first[0].ball_xy, second[0].ball_xy, equal_nan=True)
        assert [e.start_frame for e in first[1]] == [e.start_frame for e in second[1]]

    def test_a_different_seed_gives_a_different_match(self) -> None:
        assert not np.array_equal(
            SyntheticDemoMatchSource(seed=42).load()[0].ball_xy,
            SyntheticDemoMatchSource(seed=43).load()[0].ball_xy,
            equal_nan=True,
        )

    def test_coordinates_and_timestamps_are_finite(self) -> None:
        tracking, _, _ = SyntheticDemoMatchSource().load()

        assert np.all(np.isfinite(tracking.time_s))
        assert np.all(np.isfinite(tracking.home_xy))
        assert np.all(np.isfinite(tracking.away_xy))
        # The ball may legitimately be absent; nothing else may be.
        assert np.isfinite(tracking.ball_xy).mean() > 0.9

    def test_it_runs_long_enough_to_be_worth_watching(self) -> None:
        tracking, _, _ = SyntheticDemoMatchSource().load()

        # Two periods of four minutes: about a minute of wall-clock at the
        # demo's 8x, which is long enough to see the replay loop.
        assert tracking.duration_s >= 400.0
        assert tracking.n_frames >= 10_000
        assert set(np.unique(tracking.period)) == {1, 2}

    def test_it_produces_the_event_the_demo_is_about(self) -> None:
        match: SyntheticMatch = generate_synthetic_match(
            seed=42, n_periods=2, period_duration_s=240.0
        )
        assert len(match.true_entries) > 0, "no penalty-area entries to report on"


class TestPublicStartup:
    """Starting with no dataset, and with or without a model artifact."""

    def test_it_starts_with_no_data_and_no_registry(self, tmp_path: Path) -> None:
        app = create_configured_app(_public_settings(tmp_path), speed=0.0)

        with TestClient(app) as raw:
            body = ApiClient(raw).get("/ready").json()

        assert body["ready"] is True
        assert body["mode"] == "public_demo"
        assert body["data_source"] == "synthetic"

    def test_without_an_artifact_it_says_so_rather_than_pretending(self, tmp_path: Path) -> None:
        """The fallback path, proven rather than assumed.

        An image whose model failed to build still starts — that is the point of
        the fallback — but it must describe itself accurately, because the page
        renders `is_ml` directly.
        """
        app = create_configured_app(_public_settings(tmp_path), speed=0.0)

        with TestClient(app) as raw:
            body = ApiClient(raw).get("/ready").json()

        assert body["predictor"]["is_ml"] is False
        assert body["predictor"]["kind"] == "heuristic"
        assert body["predictor"]["name"] == "heuristic-fallback"

    def test_local_mode_reports_missing_data_rather_than_switching(self) -> None:
        """The failure that must never become a silent fallback.

        Serving a generated fixture to someone who asked for a real match would
        make every number on the page a fiction presented as a measurement.
        """
        settings = Settings()
        settings.paths.raw_dir = NOWHERE / "raw"

        with pytest.raises(DataValidationError) as caught:
            create_configured_app(settings, "Sample_Game_2")

        message = str(caught.value)
        assert "Sample_Game_2" in message
        assert "football-insights acquire" in message

    def test_an_unknown_match_names_the_catalogue(self) -> None:
        with pytest.raises(DataValidationError) as caught:
            create_configured_app(Settings(), "Sample_Game_9")

        assert "Sample_Game_1" in str(caught.value)
        assert SYNTHETIC_MATCH_ID in str(caught.value)


class TestReadinessSemantics:
    """What readiness answers, and when."""

    def test_it_is_ready_before_any_stream_client_connects(self, tmp_path: Path) -> None:
        """The condition a deployment health check depends on.

        The replay task starts on the first subscriber. If an idle replay counted
        as unready, the platform's health check — which runs long before any
        browser opens the page — would never pass, and it would restart a working
        container forever.
        """
        app = create_configured_app(_public_settings(tmp_path), speed=0.0)

        with TestClient(app) as raw:
            client = ApiClient(raw)
            response = client.get("/ready")
            assert response.status_code == 200
            assert response.json()["ready"] is True
            assert response.json()["replay"] == "idle"

            # Still ready on a second call, with nothing having subscribed.
            assert client.get("/ready").json()["ready"] is True

    def test_a_missing_page_is_not_ready_in_public_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_module, "_mount_demo", _no_demo_page)
        app = create_configured_app(_public_settings(tmp_path), speed=0.0)

        with TestClient(app) as raw:
            response = ApiClient(raw).get("/ready")

        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert response.json()["ui"] is False
        assert response.json()["reason"] == "demo page not built"

    def test_a_missing_page_is_fine_outside_public_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An API-only deployment is legitimate and must stay ready."""
        monkeypatch.setattr(app_module, "_mount_demo", _no_demo_page)
        settings = Settings()
        settings.paths.artifacts_dir = tmp_path
        engine, metrics = _engine(settings)
        app = create_app(settings, engine, None, metrics, data_source="metrica")

        with TestClient(app) as raw:
            body = ApiClient(raw).get("/ready").json()

        assert body["ready"] is True
        assert body["mode"] == "local"

    def test_the_payload_carries_no_paths_or_fingerprints(self, tmp_path: Path) -> None:
        """It is served unauthenticated on a public URL."""
        app = create_configured_app(_public_settings(tmp_path), speed=0.0)

        with TestClient(app) as raw:
            body = ApiClient(raw).get("/ready").text

        assert "/nonexistent" not in body
        assert str(tmp_path) not in body
        assert "fingerprint" not in body


class TestPublicSurface:
    """Which routes a public visitor can reach."""

    @staticmethod
    def _app(tmp_path: Path, *, public: bool) -> FastAPI:
        settings = _public_settings(tmp_path)
        settings.service.public_demo = public
        if not public:
            settings.replay.match_id = SYNTHETIC_MATCH_ID
        return create_configured_app(settings, speed=0.0)

    @staticmethod
    def _paths(app: FastAPI) -> set[str]:
        """Route templates this build actually published."""
        return set(app.openapi()["paths"])

    def test_mutating_routes_are_absent_from_the_schema(self, tmp_path: Path) -> None:
        app = self._app(tmp_path, public=True)
        paths = self._paths(app)

        assert "/replay/control" not in paths
        assert "/replay/match" not in paths
        # The read-only half is still there.
        assert "/replay/status" in paths
        assert "/replay/matches" in paths

    def test_mutating_routes_are_present_outside_public_mode(self, tmp_path: Path) -> None:
        app = self._app(tmp_path, public=False)
        paths = self._paths(app)

        assert "/replay/control" in paths
        assert "/replay/match" in paths

    def test_mutating_routes_cannot_be_reached(self, tmp_path: Path) -> None:
        """Not registered, so the request fails at routing.

        The exact code depends on whether the built page is mounted at `/` — a
        static mount answers an unknown POST with 405, bare routing with 404 —
        and both mean the same thing. What matters is that neither succeeds.
        """
        with TestClient(self._app(tmp_path, public=True)) as raw:
            client = ApiClient(raw)
            assert client.post("/replay/control", json={"paused": True}).status_code in {404, 405}
            assert client.post("/replay/match", json={"match": "Sample_Game_1"}).status_code in {
                404,
                405,
            }

    def test_capabilities_advertises_the_read_only_surface(self, tmp_path: Path) -> None:
        with TestClient(self._app(tmp_path, public=True)) as raw:
            body = ApiClient(raw).get("/capabilities").json()

        assert body == {
            "pipeline_controls": False,
            "replay_controls": False,
            "public_demo": True,
        }

    def test_pipeline_controls_stay_off_in_public_mode(self, tmp_path: Path) -> None:
        paths = self._paths(self._app(tmp_path, public=True))
        assert not any(p.startswith("/jobs") for p in paths)


class TestMetricCardinality:
    """Request metrics must not grow a series per URL."""

    def test_unmatched_paths_share_one_label(self, tmp_path: Path) -> None:
        settings = Settings()
        settings.paths.artifacts_dir = tmp_path
        engine, metrics = _engine(settings)
        app = create_app(settings, engine, None, metrics)

        with TestClient(app) as raw:
            client = ApiClient(raw)
            for path in ("/.env", "/wp-login.php", "/admin/../etc/passwd", "/nope"):
                client.get(path)
            exposition = client.get("/metrics").text

        assert f'endpoint="{UNMATCHED_ROUTE}"' in exposition
        for path in (".env", "wp-login", "passwd", "nope"):
            assert path not in exposition

    def test_known_routes_are_labelled_by_template(self, tmp_path: Path) -> None:
        settings = Settings()
        settings.paths.artifacts_dir = tmp_path
        engine, metrics = _engine(settings)
        app = create_app(settings, engine, None, metrics)

        with TestClient(app) as raw:
            client = ApiClient(raw)
            client.get("/health")
            exposition = client.get("/metrics").text

        assert 'endpoint="/health"' in exposition


def _no_demo_page(_app: FastAPI) -> bool:
    """Stand in for `_mount_demo` on a build where the page was never built."""
    return False


def _oversized_chunks() -> Iterator[bytes]:
    """A body streamed in pieces, larger than the cap and declaring no length.

    A generator body makes httpx use chunked transfer encoding, which is the
    case a `Content-Length` check cannot see at all.
    """
    chunk = b"x" * 64_000
    for _ in range((MAX_REQUEST_BODY_BYTES // len(chunk)) + 2):
        yield chunk


class TestPredictBounds:
    """`/predict` must not be usable to allocate arbitrary memory."""

    @staticmethod
    def _client(tmp_path: Path) -> TestClient:
        settings = Settings()
        settings.paths.artifacts_dir = tmp_path
        engine, metrics = _engine(settings)
        return TestClient(create_app(settings, engine, None, metrics))

    def test_a_valid_window_is_accepted(self, tmp_path: Path) -> None:
        window = np.zeros((1, DEFAULT_FEATURE_SPEC.n_features)).tolist()
        with self._client(tmp_path) as raw:
            assert ApiClient(raw).post("/predict", json={"window": window}).status_code == 200

    def test_too_many_timesteps_is_refused(self, tmp_path: Path) -> None:
        window = np.zeros((5000, 2)).tolist()
        with self._client(tmp_path) as raw:
            assert ApiClient(raw).post("/predict", json={"window": window}).status_code == 422

    def test_an_oversized_declared_body_is_refused(self, tmp_path: Path) -> None:
        with self._client(tmp_path) as raw:
            response = ApiClient(raw).post(
                "/predict",
                content=b"{}",
                headers={
                    "content-type": "application/json",
                    "content-length": str(MAX_REQUEST_BODY_BYTES + 1),
                },
            )
        assert response.status_code == 413

    def test_an_oversized_streamed_body_is_refused(self, tmp_path: Path) -> None:
        """The case a Content-Length check cannot catch.

        A chunked request declares no length at all, so the only limit that
        holds is one applied to the bytes as they arrive.
        """
        with self._client(tmp_path) as raw:
            response = ApiClient(raw).post(
                "/predict",
                content=_oversized_chunks(),
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 413

    def test_a_body_larger_than_it_claims_is_refused(self, tmp_path: Path) -> None:
        """A small declared length must not buy an unbounded body."""
        with self._client(tmp_path) as raw:
            response = ApiClient(raw).post(
                "/predict",
                content=_oversized_chunks(),
                headers={
                    "content-type": "application/json",
                    # Declared small, sent large: the header buys nothing.
                    "content-length": "2",
                },
            )
        assert response.status_code == 413


def _engine(settings: Settings) -> tuple[InsightEngine, Metrics]:
    """A ready engine over the generated fixture, with the fallback predictor.

    The registry comes back alongside the engine rather than hanging off it,
    because `create_app` needs the same object: a second `Metrics` would build a
    registry `GET /metrics` never reads, leaving every counter apparently at zero.
    """
    tracking, events, orientation = SyntheticDemoMatchSource(seed=5, period_duration_s=20.0).load()
    metrics = Metrics()
    engine = build_engine(
        settings,
        tracking,
        events,
        orientation,
        HeuristicPredictor(settings.model.threshold),
        metrics,
    )
    return engine, metrics


class TestArtifactLoading:
    """Which artifact the loader picks, and what it refuses."""

    def test_an_onnx_artifact_without_metadata_is_refused(self, tmp_path: Path) -> None:
        """A graph carries no schema hash, threshold or provenance.

        Serving one without its sidecar would mean inventing all three.
        """
        (tmp_path / "demo-synthetic-gru.onnx").write_bytes(b"not really a graph")
        settings = Settings()
        settings.paths.registry_dir = tmp_path
        settings.model.model_name = "demo-synthetic-gru"

        with pytest.raises(DataValidationError, match="no metadata"):
            load_predictor(settings)

    def test_a_missing_artifact_falls_back(self, tmp_path: Path) -> None:
        settings = Settings()
        settings.paths.registry_dir = tmp_path / "empty"
        settings.model.model_name = "demo-synthetic-gru"

        predictor = load_predictor(settings)

        assert predictor.metadata.is_ml is False
        assert predictor.metadata.name == "heuristic-fallback"


class TestLoopingReplay:
    """A looping demo must keep working after it wraps."""

    def test_the_player_counts_laps(self) -> None:
        tracking, _, _ = SyntheticDemoMatchSource(seed=3, period_duration_s=1.0).load()
        player = ReplayPlayer(
            match_id=SYNTHETIC_MATCH_ID,
            tracking=tracking,
            profile=Settings().fault_profile("clean"),
            seed=1,
            speed=0.0,
        )
        assert player.laps == 0

    async def test_the_first_frame_of_a_new_lap_is_accepted(self, tmp_path: Path) -> None:
        """The bug a looping demo would otherwise have.

        `stream(loop=True)` rewinds the player, but the engine's monotonic frame
        check is still at the end of the match. Without resetting it, every frame
        of every later lap is rejected as out of order — and the pitch keeps
        animating regardless, because frames are published whether or not the
        engine accepted them. The demo would look alive and never say anything
        again.

        So the assertion is specifically that the *first* frame after the wrap is
        processed, not that insights resume eventually.
        """
        settings = Settings()
        settings.paths.artifacts_dir = tmp_path
        settings.replay.loop = True

        tracking, events, orientation = SyntheticDemoMatchSource(
            seed=3, n_periods=1, period_duration_s=2.0
        ).load()
        metrics = Metrics()
        engine = build_engine(
            settings,
            tracking,
            events,
            orientation,
            HeuristicPredictor(settings.model.threshold),
            metrics,
        )
        player = ReplayPlayer(
            match_id=SYNTHETIC_MATCH_ID,
            tracking=tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=0.0,
        )
        state = AppState(settings=settings, metrics=metrics, engine=engine, player=player)
        queue = state.subscribe()  # keeps the public-mode idle pause from firing

        processed = 0
        first_of_second_lap: bool | None = None
        laps = player.laps

        async for emitted in player.stream(loop=True):
            wrapped = player.laps != laps
            if wrapped:
                laps = player.laps
                apply_restart(state, engine)

            result = engine.process(emitted.frame)
            if wrapped and first_of_second_lap is None:
                first_of_second_lap = result.frame_accepted

            processed += 1
            if first_of_second_lap is not None or processed > tracking.n_frames * 2 + 10:
                player.stop()

        assert first_of_second_lap is True, (
            "the first frame after the replay wrapped was rejected by the engine"
        )
        assert queue is not None

    async def test_without_the_reset_the_new_lap_is_rejected(self, tmp_path: Path) -> None:
        """The same loop with the reset removed, to show the check is load-bearing.

        If this ever starts passing, the engine no longer enforces frame order
        and the reset above is protecting nothing.
        """
        settings = Settings()
        settings.paths.artifacts_dir = tmp_path
        tracking, events, orientation = SyntheticDemoMatchSource(
            seed=3, n_periods=1, period_duration_s=2.0
        ).load()
        metrics = Metrics()
        engine = build_engine(
            settings,
            tracking,
            events,
            orientation,
            HeuristicPredictor(settings.model.threshold),
            metrics,
        )
        player = ReplayPlayer(
            match_id=SYNTHETIC_MATCH_ID,
            tracking=tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=0.0,
        )

        laps = player.laps
        rejected_after_wrap: bool | None = None
        async for emitted in player.stream(loop=True):
            wrapped = player.laps != laps
            result = engine.process(emitted.frame)
            if wrapped:
                # No apply_restart here: this is the failure mode, reproduced.
                rejected_after_wrap = result.prediction is None and result.attacking_team is None
                player.stop()

        assert rejected_after_wrap is True


async def _until_stopped(state: AppState) -> None:
    """Wait for the replay loop to notice it has no audience and end."""
    while state.replay_running:
        await asyncio.sleep(0.01)


class TestUnwatchedDemo:
    """What happens to a public replay when nobody is watching it."""

    @staticmethod
    def _state(tmp_path: Path, speed: float = 0.0) -> AppState:
        settings = _public_settings(tmp_path)
        engine, metrics = _engine(settings)
        tracking, _, _ = SyntheticDemoMatchSource(seed=5, period_duration_s=20.0).load()
        player = ReplayPlayer(
            match_id=SYNTHETIC_MATCH_ID,
            tracking=tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=speed,
        )
        return AppState(settings=settings, metrics=metrics, engine=engine, player=player)

    async def test_the_replay_stops_when_the_last_viewer_leaves(self, tmp_path: Path) -> None:
        state = self._state(tmp_path)
        queue = state.subscribe()
        state.ensure_replay_task()
        await asyncio.sleep(0)
        assert state.replay_running is True

        state.unsubscribe(queue)
        # The loop notices on its next frame and returns.
        await asyncio.wait_for(_until_stopped(state), timeout=10.0)
        assert state.replay_running is False

    async def test_a_second_viewer_still_gets_frames(self, tmp_path: Path) -> None:
        """The failure this replaced, reproduced as a test.

        The first attempt paused the *player* instead of stopping the task. A
        paused player never yields another frame, so the loop body that would
        have noticed a viewer returning never ran again — the demo went dark
        permanently for everyone after the first person closed their tab, while
        still reporting itself healthy and ready.
        """
        state = self._state(tmp_path)

        first = state.subscribe()
        state.ensure_replay_task()
        assert await asyncio.wait_for(first.get(), timeout=10.0) is not None
        state.unsubscribe(first)
        await asyncio.wait_for(_until_stopped(state), timeout=10.0)

        # Someone else opens the page a moment later.
        second = state.subscribe()
        state.ensure_replay_task()
        message = await asyncio.wait_for(second.get(), timeout=10.0)

        assert message.data
        state.unsubscribe(second)
        await asyncio.wait_for(_until_stopped(state), timeout=10.0)

    async def test_readiness_reports_idle_once_the_replay_has_stopped(self, tmp_path: Path) -> None:
        """The field must follow the task, not the player.

        A replay that ended because nobody was watching leaves the player
        mid-match and still marked running, so reading the player would report a
        replay that nothing is driving — and the page shows this value.
        """
        state = self._state(tmp_path)
        queue = state.subscribe()
        state.ensure_replay_task()
        await asyncio.wait_for(queue.get(), timeout=10.0)
        assert state.replay_running is True

        state.unsubscribe(queue)
        await asyncio.wait_for(_until_stopped(state), timeout=10.0)

        assert state.player is not None
        # The player is abandoned mid-match and still says it is running...
        assert state.player.status().running is True
        # ...so `/ready` reads this instead, which is what actually drives frames.
        assert state.replay_running is False

    async def test_a_local_replay_is_left_alone(self, tmp_path: Path) -> None:
        """Only public deployments do this; a local run may be watched by a CLI.

        Paced at real time so the fixture cannot simply run to completion inside
        the test — otherwise a loop that ended on its own would look like the
        unwatched path firing when it had not.
        """
        state = self._state(tmp_path, speed=1.0)
        state.settings.service.public_demo = False
        queue = state.subscribe()
        state.ensure_replay_task()
        assert await asyncio.wait_for(queue.get(), timeout=10.0) is not None

        state.unsubscribe(queue)
        assert state.should_stop_unwatched() is False
        for _ in range(50):
            await asyncio.sleep(0)

        assert state.replay_running is True
        await state.stop_replay_task()


class TestStreamEndSignal:
    """The end marker is a typed value, not a substring of the payload."""

    def test_the_type_survives_reserialising_the_payload(self) -> None:
        """The coupling the substring search had, and this does not.

        The old check searched the serialised JSON for `'"type": "end"'`, which
        held only while `json.dumps` was called with its default separators.
        Compacting the payload would have stopped it matching and left every
        client waiting on a finished stream. Here the same compaction changes
        nothing a consumer reads.
        """
        from football_insights.serving.messages import StreamMessageType, stream_message

        message = stream_message(StreamMessageType.END, {"frames": 12})
        compact = json.dumps(json.loads(message.data), separators=(",", ":"))

        assert '"type": "end"' not in compact  # the old search would miss this
        assert message.type is StreamMessageType.END  # this does not

    def test_every_message_kind_is_distinguishable_without_parsing(self) -> None:
        from football_insights.serving.messages import StreamMessageType, stream_message

        kinds = [stream_message(kind, {}).type for kind in StreamMessageType]

        assert kinds == list(StreamMessageType)
        assert sum(k is StreamMessageType.END for k in kinds) == 1

    def test_the_wire_format_is_unchanged(self) -> None:
        """Existing clients switch on the JSON `type`; that must still be there."""
        from football_insights.serving.messages import StreamMessageType, stream_message

        message = stream_message(StreamMessageType.END, {"frames": 12})

        assert message.data == json.dumps({"type": "end", "payload": {"frames": 12}})
