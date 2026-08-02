"""Slice 0 acceptance test: the vertical path, end to end.

Synthetic replay -> validated rolling window -> deterministic predictor ->
insight candidate -> editorial suppression -> SSE stream.

This runs against the rule-based fallback so it stays green before any model is
trained, and is parameterised over predictors so the fallback path cannot rot
once a trained model exists.
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from football_insights.config import Settings
from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
from football_insights.domain import Team
from football_insights.insight.templates import is_hedged
from football_insights.insight.types import Insight, InsightKind, SuppressionReason
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.replay.player import ReplayPlayer
from football_insights.serving.app import create_app
from football_insights.serving.engine import InsightEngine
from football_insights.serving.metrics import Metrics
from football_insights.serving.state import AppState
from football_insights.serving.stream import SUPPRESSION_ROLLUP_FRAMES
from football_insights.types import JsonDict
from tests.support import ApiClient

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
    insights: list[Insight] = []
    for frame in match.tracking.iter_frames():
        result = engine.process(frame)
        if result.outcome is not None and result.outcome.insight is not None:
            insights.append(result.outcome.insight)
    return insights, metrics, engine


def _stub_insight() -> Insight:
    """A recognisable insight to plant in history, to prove a restart clears it."""
    return Insight(
        kind=InsightKind.BUILDING_THREAT,
        headline="from a previous replay",
        detail="",
        probability=0.5,
        match_time_s=1.0,
        period=1,
        attacking_team="home",
        model_name="stub",
        model_version="0.0.0",
        is_ml=False,
    )


@pytest.fixture(scope="module")
def match() -> SyntheticMatch:
    return generate_synthetic_match(seed=5, period_duration_s=180.0)


@pytest.fixture
def settings() -> Settings:
    return Settings()


class TestReadiness:
    """Criterion 1: liveness and readiness are distinct and honest."""

    def test_health_is_up_without_a_model(self, match: SyntheticMatch, settings: Settings) -> None:
        metrics = Metrics()
        engine = build_engine(match, settings, None, metrics)
        client = ApiClient(TestClient(create_app(settings, engine, None, metrics)))
        assert client.get("/health").status_code == 200

    def test_ready_is_false_without_a_model(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        metrics = Metrics()
        engine = build_engine(match, settings, None, metrics)
        client = ApiClient(TestClient(create_app(settings, engine, None, metrics)))
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert "predictor" in response.json()["reason"]

    def test_model_endpoint_refuses_without_a_model(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        metrics = Metrics()
        engine = build_engine(match, settings, None, metrics)
        client = ApiClient(TestClient(create_app(settings, engine, None, metrics)))
        assert client.get("/model").status_code == 503

    def test_ready_is_true_with_the_fallback(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        metrics = Metrics()
        engine = build_engine(match, settings, HeuristicPredictor(), metrics)
        client = ApiClient(TestClient(create_app(settings, engine, None, metrics)))
        assert client.get("/ready").json()["ready"] is True


class TestStream:
    """Criterion 2: the SSE stream carries frames from the replay."""

    def test_stream_yields_frames(self, match: SyntheticMatch, settings: Settings) -> None:
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
        client = ApiClient(TestClient(create_app(settings, engine, player, metrics)))
        seen: list[JsonDict] = []
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
        assert {
            "period",
            "match_time_s",
            "home",
            "away",
            "ball",
            # The demo reads all three. `attacking_right` is what makes the
            # target penalty area — the thing the model is predicting entry into
            # — drawable at all.
            "attacking_team",
            "attacking_right",
            "suppression",
        } <= set(payload)
        assert len(payload["home"]) == len(short.tracking.home_players)

    def test_frames_carry_attacking_direction(self, settings: Settings) -> None:
        """Direction is a boolean in canonical coordinates, or honestly absent.

        A frame whose possession is unresolved reports ``None`` for both fields
        rather than guessing a side; "unknown" and "attacking left" are
        different statements and the wire keeps them apart.
        """
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
        client = ApiClient(TestClient(create_app(settings, engine, player, metrics)))
        payloads: list[JsonDict] = []
        with client.stream("GET", "/insights/stream") as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    message = json.loads(line[5:].strip())
                    if message["type"] == "frame":
                        payloads.append(message["payload"])
                if len(payloads) >= 40:
                    break

        for payload in payloads:
            team, right = payload["attacking_team"], payload["attacking_right"]
            assert (team is None) == (right is None), "team and direction must agree on absence"
            assert team in {None, "home", "away"}
            assert right in {None, True, False}
        assert any(p["attacking_team"] is not None for p in payloads), (
            "no frame resolved possession, so direction was never exercised"
        )

    def test_suppression_rollups_are_exact_and_typed(self, settings: Settings) -> None:
        """Rollups are the only honest source of suppression totals.

        Frames are published at half the source rate, so totals accumulated from
        the frame stream would be a sample presented as a total. The rollup is
        computed over every reviewed frame, which is why the counts must never
        exceed the frames they describe — and why every key has to be a value
        from the closed :class:`SuppressionReason` enum, the same guarantee the
        Prometheus labels carry.
        """
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
        client = ApiClient(TestClient(create_app(settings, engine, player, metrics)))
        rollups: list[JsonDict] = []
        with client.stream("GET", "/insights/stream") as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    message = json.loads(line[5:].strip())
                    if message["type"] == "suppression":
                        rollups.append(message["payload"])
                if len(rollups) >= 3:
                    break

        assert rollups, "no suppression rollup was published"
        valid = {r.value for r in SuppressionReason}
        for payload in rollups:
            assert payload["frames"] == SUPPRESSION_ROLLUP_FRAMES
            counts: dict[str, int] = payload["counts"]
            assert set(counts) <= valid, f"unknown suppression reason on the wire: {set(counts)}"
            assert all(v > 0 for v in counts.values()), "zero counts should be omitted, not sent"
            assert sum(counts.values()) + payload["emitted"] <= payload["frames"]

    def test_replay_emits_every_source_frame_when_clean(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        player = ReplayPlayer(
            match_id="synthetic",
            tracking=match.tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=0.0,
        )
        assert player.total_frames == match.tracking.n_frames
        assert [e.frame.frame for e in player.emitted] == list(match.tracking.frame)


class TestReplayRestart:
    """Restarting must rebuild engine state, not merely rewind the tape.

    Driven against the replay loop directly rather than through ``TestClient``,
    because the interesting moment is the one where a control request lands
    while the loop is already holding a frame taken from the old position — and
    a blocking stream read gives a test no way to be there.
    """

    @staticmethod
    def _state(settings: Settings) -> tuple[AppState, ReplayPlayer, Metrics]:
        # Long enough that the generator actually emits events: below about 20 s
        # it produces none at all, possession never resolves, and every frame
        # yields no editorial outcome — which would make the assertions below
        # pass for the wrong reason. `_reviewed` guards that directly.
        short = generate_synthetic_match(seed=5, period_duration_s=40.0)
        metrics = Metrics()
        engine = build_engine(short, settings, HeuristicPredictor(), metrics)
        player = ReplayPlayer(
            match_id="synthetic",
            tracking=short.tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=50.0,
        )
        state = AppState(settings=settings, metrics=metrics, engine=engine, player=player)
        return state, player, metrics

    @staticmethod
    def _rollups(messages: list[JsonDict]) -> list[JsonDict]:
        return [m["payload"] for m in messages if m["type"] == "suppression"]

    @staticmethod
    def _reviewed(rollup: JsonDict) -> int:
        """Editorial decisions one rollup actually describes."""
        counts: dict[str, int] = rollup["counts"]
        return int(rollup["emitted"]) + sum(counts.values())

    @staticmethod
    async def _drain(queue: asyncio.Queue[str], count: int) -> list[JsonDict]:
        out: list[JsonDict] = []
        while len(out) < count:
            raw = await asyncio.wait_for(queue.get(), timeout=10.0)
            out.append(json.loads(raw))
        return out

    async def test_restart_rebuilds_engine_state(self, settings: Settings) -> None:
        """A held pre-restart frame must never reach the rewound engine.

        If it did, the engine's monotonic frame check would sit ahead of
        everything about to be replayed and reject all of it as out of order —
        while the pitch kept animating, because frames are published whether or
        not the engine accepted them. The tell is that no frame after the
        restart produces an editorial outcome at all, so the rollups that follow
        describe nothing.
        """
        state, player, _ = self._state(settings)
        queue = state.subscribe()
        state.ensure_replay_task()
        try:
            before = await self._drain(queue, 120)
            player.reset()
            state.request_restart()
            after = await self._drain(queue, 120)
        finally:
            player.stop()

        baseline = self._rollups(before)
        assert baseline and any(self._reviewed(r) > 0 for r in baseline), (
            "the replay scored nothing even before restarting, so the assertion below is vacuous"
        )

        markers = [m for m in after if m["type"] == "restart"]
        assert len(markers) == 1, f"expected exactly one restart marker, saw {len(markers)}"

        # Scoring has to be *sustained*, which is why several rollups are checked
        # rather than one. A rewound replay whose engine kept its old frame id
        # rejects everything until it catches back up, but the single held frame
        # — taken from the old position, so still monotonic — is accepted and
        # scored on the way past. That lone decision lands in the first rollup
        # and makes a one-rollup assertion pass while the restarted replay is in
        # fact dead for the next two hundred frames.
        tail = self._rollups(after[after.index(markers[0]) :])
        assert len(tail) >= 3, "too few rollups followed the restart to judge"
        for position, rollup in enumerate(tail[:3]):
            assert self._reviewed(rollup) > 0, (
                f"rollup {position} after the restart describes no decision at all: "
                "frames are being rejected by an engine that was not rebuilt"
            )

    async def test_restart_clears_the_insight_history(self, settings: Settings) -> None:
        """`GET /insights` must not serve insights from a replay that no longer exists."""
        state, player, _ = self._state(settings)
        queue = state.subscribe()
        sentinel = _stub_insight()
        state.recent_insights.append(sentinel)
        state.ensure_replay_task()
        try:
            await self._drain(queue, 5)
            player.reset()
            state.request_restart()
            await self._drain(queue, 40)
        finally:
            player.stop()

        assert sentinel not in state.recent_insights


class TestEngineDirection:
    """Direction is resolved once, in the engine, and travels with the result.

    It lives on ``EngineResult`` rather than on ``Prediction`` because the two
    have different lifetimes: when no predictor is loaded there is no prediction
    at all, but which way the attacking team is playing is still known and still
    worth showing. A presentation layer deriving it instead would have to
    reimplement the half-swap the engine already applies to the ball.
    """

    @staticmethod
    def _observed(
        match: SyntheticMatch, settings: Settings, predictor: object | None
    ) -> set[tuple[int, str, float]]:
        """Every (period, team, sign) the engine reported over a whole match."""
        engine = build_engine(match, settings, predictor, Metrics())
        seen: set[tuple[int, str, float]] = set()
        for frame in match.tracking.iter_frames():
            result = engine.process(frame)
            if result.attacking_team is None or result.attacking_sign is None:
                continue
            seen.add((frame.period, result.attacking_team, result.attacking_sign))
        return seen

    def test_direction_matches_the_orientation_table(self, settings: Settings) -> None:
        short = generate_synthetic_match(seed=5, period_duration_s=20.0)
        observed = self._observed(short, settings, HeuristicPredictor())
        assert observed, "no frame resolved possession"
        for period, team, sign in observed:
            expected = short.orientation.direction(period, Team(team)).sign
            assert sign == expected, (
                f"period {period} {team} reported {sign}, table says {expected}"
            )

    def test_direction_flips_at_the_period_change(self, settings: Settings) -> None:
        """Teams change ends at half time, so the sign must invert with them."""
        short = generate_synthetic_match(seed=5, period_duration_s=20.0)
        observed = self._observed(short, settings, HeuristicPredictor())
        by_period = {(p, t): s for p, t, s in observed}
        flipped = [
            (team, by_period[(1, team)], by_period[(2, team)])
            for team in ("home", "away")
            if (1, team) in by_period and (2, team) in by_period
        ]
        assert flipped, "no team was seen attacking in both periods"
        for team, first, second in flipped:
            assert first == -second, f"{team} attacked the same way in both periods"

    def test_direction_survives_an_unavailable_model(self, settings: Settings) -> None:
        """A viewer can still be told which way play is running while the model is down."""
        short = generate_synthetic_match(seed=5, period_duration_s=20.0)
        observed = self._observed(short, settings, None)
        assert observed, "direction was lost when no predictor was loaded"


class TestInsightWording:
    """Criterion 3: nothing reaches a viewer as an unqualified claim."""

    def test_every_insight_is_hedged_and_attributed(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        insights, _, _ = run_replay(match, settings, HeuristicPredictor())
        assert insights, "fixture produced no insights to check"
        for insight in insights:
            assert is_hedged(insight.headline), f"unhedged headline: {insight.headline!r}"
            assert insight.model_version
            assert insight.model_name == "heuristic-fallback"
            assert insight.is_ml is False

    def test_fallback_is_flagged_as_not_ml_on_the_api(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        metrics = Metrics()
        engine = build_engine(match, settings, HeuristicPredictor(), metrics)
        client = ApiClient(TestClient(create_app(settings, engine, None, metrics)))
        body = client.get("/model").json()
        assert body["is_ml"] is False
        assert body["kind"] == "heuristic"


class TestInvalidWindowsProduceNothing:
    """Criterion 4: structurally invalid windows emit no insight."""

    def test_gap_suppresses_and_is_counted(self, settings: Settings) -> None:
        broken = generate_synthetic_match(seed=5, period_duration_s=180.0)
        # Blank the ball for a contiguous ten seconds mid-match.
        rate = int(broken.frame_rate)
        start, stop = 60 * rate, 70 * rate
        broken.tracking.ball_xy[start:stop] = np.nan

        insights, metrics, _ = run_replay(broken, settings, HeuristicPredictor())
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

    def test_no_model_means_no_insight_but_a_reason(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        insights, metrics, _ = run_replay(match, settings, None)
        assert insights == []
        snapshot = metrics.snapshot()
        key = 'fi_insight_suppressed_total{reason="model_unavailable"}'
        assert snapshot.get(key, 0) > 0, "a missing model must be reported, not silent"


class TestDeterminism:
    """Criterion 5: identical inputs give identical output."""

    def test_two_runs_produce_identical_insights(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
        first, _, _ = run_replay(match, settings, HeuristicPredictor())
        second, _, _ = run_replay(match, settings, HeuristicPredictor())
        assert [i.to_dict() for i in first] == [i.to_dict() for i in second]

    def test_predictor_is_deterministic(self, match: SyntheticMatch, settings: Settings) -> None:
        predictor = HeuristicPredictor()
        window = np.random.default_rng(0).normal(size=(4, 50, 39)).astype(np.float32)
        np.testing.assert_array_equal(
            predictor.predict_proba(window), predictor.predict_proba(window)
        )


class TestMetricsExposure:
    """Criterion 6: both metric namespaces are exposed and populated."""

    def test_both_namespaces_present(self, match: SyntheticMatch, settings: Settings) -> None:
        _, metrics, _ = run_replay(match, settings, HeuristicPredictor())
        client = ApiClient(TestClient(create_app(settings, None, None, metrics)))
        body = client.get("/metrics").text
        assert "fi_model_predictions_total" in body
        assert "fi_model_inference_latency_seconds" in body
        assert "fi_insight_candidates_total" in body
        assert "fi_insight_suppressed_total" in body

    def test_suppression_reasons_are_recorded(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
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

    def test_model_and_editorial_counts_are_independent(
        self, match: SyntheticMatch, settings: Settings
    ) -> None:
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
