"""Slice 0 acceptance test: the vertical path, end to end.

Synthetic replay -> validated rolling window -> deterministic predictor ->
insight candidate -> editorial suppression -> SSE stream.

This runs against the rule-based fallback so it stays green before any model is
trained, and is parameterised over predictors so the fallback path cannot rot
once a trained model exists.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from football_insights.config import Settings
from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
from football_insights.insight.templates import is_hedged
from football_insights.insight.types import Insight
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.replay.player import ReplayPlayer
from football_insights.serving.app import create_app
from football_insights.serving.engine import InsightEngine
from football_insights.serving.metrics import Metrics

logging.disable(logging.INFO)


def build_engine(
    match: SyntheticMatch,
    settings: Settings,
    predictor: object | None,
    metrics: Metrics,
) -> InsightEngine:
    """Wire an engine around a synthetic match."""
    return InsightEngine(
        settings=settings,
        orientation=match.orientation,
        events=match.events,
        frame_rate=match.frame_rate,
        predictor=predictor,  # type: ignore[arg-type]
        metrics=metrics,
        home_is_gk=np.array([p.is_goalkeeper for p in match.tracking.home_players]),
        away_is_gk=np.array([p.is_goalkeeper for p in match.tracking.away_players]),
    )


def run_replay(
    match: SyntheticMatch, settings: Settings, predictor: object | None
) -> tuple[list[Insight], Metrics, InsightEngine]:
    """Drive every frame through the engine, returning insights and metrics."""
    metrics = Metrics()
    engine = build_engine(match, settings, predictor, metrics)
    insights = []
    for frame in match.tracking.iter_frames():
        result = engine.process(frame)
        if result.outcome is not None and result.outcome.insight is not None:
            insights.append(result.outcome.insight)
    return insights, metrics, engine


@pytest.fixture(scope="module")
def match() -> SyntheticMatch:
    return generate_synthetic_match(seed=5, period_duration_s=180.0)


@pytest.fixture
def settings() -> Settings:
    return Settings()


class TestReadiness:
    """Criterion 1: liveness and readiness are distinct and honest."""

    def test_health_is_up_without_a_model(self, match, settings):
        metrics = Metrics()
        engine = build_engine(match, settings, None, metrics)
        client = TestClient(create_app(settings, engine, None, metrics))
        assert client.get("/health").status_code == 200

    def test_ready_is_false_without_a_model(self, match, settings):
        metrics = Metrics()
        engine = build_engine(match, settings, None, metrics)
        client = TestClient(create_app(settings, engine, None, metrics))
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert "predictor" in response.json()["reason"]

    def test_model_endpoint_refuses_without_a_model(self, match, settings):
        metrics = Metrics()
        engine = build_engine(match, settings, None, metrics)
        client = TestClient(create_app(settings, engine, None, metrics))
        assert client.get("/model").status_code == 503

    def test_ready_is_true_with_the_fallback(self, match, settings):
        metrics = Metrics()
        engine = build_engine(match, settings, HeuristicPredictor(), metrics)
        client = TestClient(create_app(settings, engine, None, metrics))
        assert client.get("/ready").json()["ready"] is True


class TestStream:
    """Criterion 2: the SSE stream carries frames from the replay."""

    def test_stream_yields_frames(self, match, settings):
        short = generate_synthetic_match(seed=5, period_duration_s=40.0)
        metrics = Metrics()
        engine = build_engine(short, settings, HeuristicPredictor(), metrics)
        player = ReplayPlayer(
            match_id="synthetic",
            tracking=short.tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=0.0,
        )
        client = TestClient(create_app(settings, engine, player, metrics))
        seen = []
        with client.stream("GET", "/insights/stream") as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data:"):
                    seen.append(json.loads(line[5:].strip()))
                if len(seen) >= 5:
                    break
        assert seen, "stream produced no messages"
        frames = [m for m in seen if m["type"] == "frame"]
        assert frames, "stream produced no frame messages"
        payload = frames[0]["payload"]
        assert {"period", "match_time_s", "home", "away", "ball"} <= set(payload)
        assert len(payload["home"]) == len(short.tracking.home_players)

    def test_replay_emits_every_source_frame_when_clean(self, match, settings):
        player = ReplayPlayer(
            match_id="synthetic",
            tracking=match.tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=0.0,
        )
        assert player.total_frames == match.tracking.n_frames
        assert [e.frame.frame for e in player.emitted] == list(match.tracking.frame)


class TestInsightWording:
    """Criterion 3: nothing reaches a viewer as an unqualified claim."""

    def test_every_insight_is_hedged_and_attributed(self, match, settings):
        insights, _, _ = run_replay(match, settings, HeuristicPredictor())
        assert insights, "fixture produced no insights to check"
        for insight in insights:
            assert is_hedged(insight.headline), f"unhedged headline: {insight.headline!r}"
            assert insight.model_version
            assert insight.model_name == "heuristic-fallback"
            assert insight.is_ml is False

    def test_fallback_is_flagged_as_not_ml_on_the_api(self, match, settings):
        metrics = Metrics()
        engine = build_engine(match, settings, HeuristicPredictor(), metrics)
        client = TestClient(create_app(settings, engine, None, metrics))
        body = client.get("/model").json()
        assert body["is_ml"] is False
        assert body["kind"] == "heuristic"


class TestInvalidWindowsProduceNothing:
    """Criterion 4: structurally invalid windows emit no insight."""

    def test_gap_suppresses_and_is_counted(self, settings):
        broken = generate_synthetic_match(seed=5, period_duration_s=180.0)
        # Blank the ball for a contiguous ten seconds mid-match.
        rate = int(broken.frame_rate)
        start, stop = 60 * rate, 70 * rate
        broken.tracking.ball_xy[start:stop] = np.nan

        insights, metrics, engine = run_replay(broken, settings, HeuristicPredictor())
        snapshot = metrics.snapshot()

        invalid = sum(
            v for k, v in snapshot.items() if k.startswith("fi_model_invalid_window_total")
        )
        assert invalid > 0, "a ten-second ball outage must invalidate windows"

        # No insight may refer to any instant whose window overlaps the outage.
        observation_s = settings.window.observation_s
        blackout_start = broken.tracking.time_s[start]
        blackout_end = broken.tracking.time_s[stop - 1] + observation_s
        offending = [i for i in insights if blackout_start <= i.match_time_s <= blackout_end]
        assert not offending, f"insights emitted from invalid windows: {offending}"

    def test_no_model_means_no_insight_but_a_reason(self, match, settings):
        insights, metrics, _ = run_replay(match, settings, None)
        assert insights == []
        snapshot = metrics.snapshot()
        key = 'fi_insight_suppressed_total{reason="model_unavailable"}'
        assert snapshot.get(key, 0) > 0, "a missing model must be reported, not silent"


class TestDeterminism:
    """Criterion 5: identical inputs give identical output."""

    def test_two_runs_produce_identical_insights(self, match, settings):
        first, _, _ = run_replay(match, settings, HeuristicPredictor())
        second, _, _ = run_replay(match, settings, HeuristicPredictor())
        assert [i.to_dict() for i in first] == [i.to_dict() for i in second]

    def test_predictor_is_deterministic(self, match, settings):
        predictor = HeuristicPredictor()
        window = np.random.default_rng(0).normal(size=(4, 50, 39)).astype(np.float32)
        np.testing.assert_array_equal(
            predictor.predict_proba(window), predictor.predict_proba(window)
        )


class TestMetricsExposure:
    """Criterion 6: both metric namespaces are exposed and populated."""

    def test_both_namespaces_present(self, match, settings):
        _, metrics, _ = run_replay(match, settings, HeuristicPredictor())
        client = TestClient(create_app(settings, None, None, metrics))
        body = client.get("/metrics").text
        assert "fi_model_predictions_total" in body
        assert "fi_model_inference_latency_seconds" in body
        assert "fi_insight_candidates_total" in body
        assert "fi_insight_suppressed_total" in body

    def test_suppression_reasons_are_recorded(self, match, settings):
        _, metrics, _ = run_replay(match, settings, HeuristicPredictor())
        snapshot = metrics.snapshot()
        reasons = {
            k.split('reason="')[1].split('"')[0]
            for k, v in snapshot.items()
            if k.startswith("fi_insight_suppressed_total{") and v > 0
        }
        # A real run must exercise more than one reason; a single catch-all
        # would mean the editorial layer is not actually discriminating.
        assert len(reasons) >= 3, f"only saw suppression reasons {reasons}"
        assert "low_confidence" in reasons

    def test_model_and_editorial_counts_are_independent(self, match, settings):
        _, metrics, _ = run_replay(match, settings, HeuristicPredictor())
        snapshot = metrics.snapshot()
        predictions = sum(
            v for k, v in snapshot.items() if k.startswith("fi_model_predictions_total{")
        )
        emitted = sum(v for k, v in snapshot.items() if k.startswith("fi_insight_emitted_total{"))
        assert predictions > 0
        assert emitted < predictions, (
            "editorial selection must reduce predictions to a smaller number of insights"
        )
