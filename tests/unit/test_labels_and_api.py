"""Label construction, episode grouping, evaluation and API contracts."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from football_insights.config import EpisodeSettings, Settings
from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
from football_insights.domain import Team
from football_insights.features.spec import DEFAULT_FEATURE_SPEC as SPEC
from football_insights.features.spec import FeatureSpec
from football_insights.labels.box_entry import (
    LabelledWindows,
    build_labels,
    merge_episodes,
    possession_clusters,
)
from football_insights.models.baseline import summarise_windows
from football_insights.models.evaluate import (
    calibration_bins,
    episode_metrics,
    group_alarms,
    window_metrics,
)
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.replay.player import ReplayPlayer
from football_insights.serving.app import create_app
from football_insights.serving.engine import InsightEngine
from football_insights.serving.metrics import Metrics
from football_insights.types import JsonDict
from tests.support import ApiClient, approx

logging.disable(logging.INFO)


@pytest.fixture(scope="module")
def match() -> SyntheticMatch:
    return generate_synthetic_match(seed=23, period_duration_s=240.0)


@pytest.fixture
def settings() -> Settings:
    return Settings()


class TestEpisodeGrouping:
    def test_close_entries_merge(self) -> None:
        entries = [(10.0, Team.HOME, 0), (14.0, Team.HOME, 1), (40.0, Team.HOME, 2)]
        periods = np.ones(3, dtype=np.int16)
        episodes = merge_episodes(entries, "m", periods, merge_gap_s=10.0)
        assert len(episodes) == 2
        assert episodes[0].n_entries == 2
        assert episodes[0].entry_time_s == 10.0

    def test_different_teams_never_merge(self) -> None:
        entries = [(10.0, Team.HOME, 0), (12.0, Team.AWAY, 1)]
        episodes = merge_episodes(entries, "m", np.ones(2, dtype=np.int16), merge_gap_s=30.0)
        assert len(episodes) == 2

    def test_merge_gap_changes_the_count(self) -> None:
        entries = [(0.0, Team.HOME, 0), (8.0, Team.HOME, 1), (16.0, Team.HOME, 2)]
        periods = np.ones(3, dtype=np.int16)
        assert len(merge_episodes(entries, "m", periods, 5.0)) == 3
        # Merging is measured from the episode's first entry, so a 10 s gap
        # absorbs the entry at 8 s but not the one at 16 s. That bound is
        # deliberate: chaining from the previous entry would let a run of
        # entries merge without limit.
        assert len(merge_episodes(entries, "m", periods, 10.0)) == 2
        assert len(merge_episodes(entries, "m", periods, 20.0)) == 1

    def test_clusters_track_possession_changes(self) -> None:
        possession = np.array([0, 0, 0, 1, 1, 0, 0])
        clusters = possession_clusters(possession)
        assert clusters.tolist() == [0, 0, 0, 1, 1, 2, 2]


class TestLabelConstruction:
    @pytest.fixture(scope="class")
    @staticmethod
    def labelled() -> tuple[SyntheticMatch, LabelledWindows]:
        m = generate_synthetic_match(seed=23, period_duration_s=240.0)
        settings = Settings()
        possession = np.zeros(m.tracking.n_frames, dtype=np.int8)
        # Alternate possession in long blocks so clusters are meaningful.
        block = int(20 * m.frame_rate)
        for start in range(0, m.tracking.n_frames, block * 2):
            possession[start + block : start + 2 * block] = 1
        dead = np.zeros(m.tracking.n_frames, dtype=bool)
        return m, build_labels(
            match_id="synthetic",
            ball_xy=m.tracking.ball_xy,
            times_s=m.tracking.time_s,
            periods=m.tracking.period,
            possession_team=possession,
            dead_ball=dead,
            orientation=m.orientation,
            window=settings.window,
            episode_settings=settings.episode,
            frame_rate=m.frame_rate,
        )

    def test_labels_are_binary_and_aligned(
        self, labelled: tuple[SyntheticMatch, LabelledWindows]
    ) -> None:
        _, result = labelled
        assert set(np.unique(result.label)) <= {0, 1}
        assert len(result.end_index) == len(result.label) == len(result.time_s)

    def test_no_window_starts_before_enough_history(
        self, labelled: tuple[SyntheticMatch, LabelledWindows], settings: Settings
    ) -> None:
        m, result = labelled
        observation = int(settings.window.observation_s * m.frame_rate)
        for end in result.end_index:
            period = m.tracking.period[end]
            first = int(np.flatnonzero(m.tracking.period == period)[0])
            assert end - first >= observation

    def test_positives_have_an_episode_and_negatives_do_not(
        self, labelled: tuple[SyntheticMatch, LabelledWindows]
    ) -> None:
        _, result = labelled
        assert np.all(result.episode_id[result.label == 1] >= 0)
        assert np.all(result.episode_id[result.label == 0] == -1)

    def test_label_looks_strictly_forward(
        self, labelled: tuple[SyntheticMatch, LabelledWindows], settings: Settings
    ) -> None:
        """No sample may be labelled from an entry at or before its own instant."""
        _, result = labelled
        times = np.array([e.entry_time_s for e in result.episodes])
        for k in np.flatnonzero(result.label == 1):
            entry = times[result.episode_id[k]]
            assert entry > result.time_s[k]
            assert entry <= result.time_s[k] + settings.window.horizon_s

    def test_windows_already_in_the_box_are_excluded(
        self, labelled: tuple[SyntheticMatch, LabelledWindows]
    ) -> None:
        m, result = labelled
        from football_insights.features.frame_features import box_entry_mask

        for k in range(0, len(result), 25):
            end = int(result.end_index[k])
            team = Team.HOME if result.attacking_team[k] == 0 else Team.AWAY
            sign = m.orientation.direction(int(result.period[k]), team).sign
            assert not bool(box_entry_mask(m.tracking.ball_xy[end : end + 1], sign)[0])

    def test_horizon_changes_the_positive_rate(self, settings: Settings) -> None:
        m = generate_synthetic_match(seed=23, period_duration_s=240.0)
        possession = np.zeros(m.tracking.n_frames, dtype=np.int8)
        dead = np.zeros(m.tracking.n_frames, dtype=bool)
        rates: list[float] = []
        for horizon in (2.0, 10.0):
            window = settings.window.model_copy(update={"horizon_s": horizon})
            result = build_labels(
                match_id="m",
                ball_xy=m.tracking.ball_xy,
                times_s=m.tracking.time_s,
                periods=m.tracking.period,
                possession_team=possession,
                dead_ball=dead,
                orientation=m.orientation,
                window=window,
                episode_settings=settings.episode,
                frame_rate=m.frame_rate,
            )
            rates.append(result.positive_rate)
        assert rates[1] > rates[0], "a longer horizon must capture more positives"


class TestEvaluation:
    def test_precision_is_zero_when_nothing_fires(self) -> None:
        y = np.array([0, 1, 0, 1])
        metrics = window_metrics(y, np.zeros(4), threshold=0.5)
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0

    def test_perfect_prediction_scores_one(self) -> None:
        y = np.array([0, 1, 0, 1])
        metrics = window_metrics(y, y.astype(float), threshold=0.5)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_alarms_bridge_short_gaps(self) -> None:
        times = np.arange(0, 10, 0.5)
        teams = np.zeros(len(times), dtype=int)
        probs = np.zeros(len(times))
        probs[[2, 3, 5, 6]] = 0.9  # one-slot gap at index 4
        assert len(group_alarms(times, teams, probs, 0.5, bridge_gap_s=2.0)) == 1
        # The bridge must cover at least the sampling stride for consecutive
        # windows to join at all; below that every window is its own alarm.
        assert len(group_alarms(times, teams, probs, 0.5, bridge_gap_s=0.5)) == 2
        assert len(group_alarms(times, teams, probs, 0.5, bridge_gap_s=0.0)) == 4

    def test_alarms_never_span_two_teams(self) -> None:
        times = np.array([0.0, 0.5, 1.0, 1.5])
        teams = np.array([0, 0, 1, 1])
        probs = np.full(4, 0.9)
        assert len(group_alarms(times, teams, probs, 0.5, bridge_gap_s=5.0)) == 2

    def test_detection_requires_the_lead_up_window(self) -> None:
        times = np.arange(0, 30, 0.5)
        teams = np.zeros(len(times), dtype=int)
        probs = np.zeros(len(times))
        probs[(times >= 8.0) & (times <= 9.5)] = 0.9
        metrics = episode_metrics(
            times_s=times,
            teams=teams,
            y_prob=probs,
            episode_times=np.array([10.0]),
            episode_teams=np.array([0]),
            threshold=0.5,
            horizon_s=5.0,
            bridge_gap_s=2.0,
            minutes=90.0,
        )
        assert metrics.detected == 1
        assert metrics.recall == 1.0
        assert metrics.median_warning_s == approx(2.0)

    def test_alarm_far_from_any_entry_is_a_false_alarm(self) -> None:
        times = np.arange(0, 60, 0.5)
        teams = np.zeros(len(times), dtype=int)
        probs = np.zeros(len(times))
        probs[(times >= 2.0) & (times <= 3.0)] = 0.9
        metrics = episode_metrics(
            times_s=times,
            teams=teams,
            y_prob=probs,
            episode_times=np.array([50.0]),
            episode_teams=np.array([0]),
            threshold=0.5,
            horizon_s=5.0,
            bridge_gap_s=2.0,
            minutes=90.0,
        )
        assert metrics.detected == 0
        assert metrics.false_alarms == 1

    def test_calibration_bins_report_observed_rates(self) -> None:
        probs = np.concatenate([np.full(50, 0.05), np.full(50, 0.95)])
        labels = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
        bins = calibration_bins(labels, probs)
        assert len(bins) == 2
        assert bins[0]["observed_rate"] == 0.0
        assert bins[-1]["observed_rate"] == 1.0


class TestFeatureContract:
    def test_schema_hash_is_stable(self) -> None:
        assert FeatureSpec().schema_hash == SPEC.schema_hash

    def test_reordering_changes_the_hash(self) -> None:
        reordered = FeatureSpec(names=tuple(reversed(SPEC.names)))
        assert reordered.schema_hash != SPEC.schema_hash

    def test_duplicate_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            FeatureSpec(names=("a", "b", "a"))

    def test_summaries_have_four_blocks(self) -> None:
        windows = np.random.default_rng(0).normal(size=(7, 50, SPEC.n_features))
        assert summarise_windows(windows).shape == (7, 4 * SPEC.n_features)


class TestApiContracts:
    @pytest.fixture
    def client(self, match: SyntheticMatch, settings: Settings) -> ApiClient:
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
            match_id="synthetic",
            tracking=match.tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=0.0,
        )
        return ApiClient(TestClient(create_app(settings, engine, player, metrics)))

    def test_health_and_model(self, client: ApiClient) -> None:
        assert client.get("/health").json()["status"] == "ok"
        body = client.get("/model").json()
        assert body["schema_matches"] is True
        assert body["running_feature_schema"] == SPEC.schema_hash

    def test_predict_accepts_a_valid_window(self, client: ApiClient) -> None:
        window = np.zeros((50, SPEC.n_features)).tolist()
        body = client.post("/predict", json={"window": window}).json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["is_ml"] is False

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            ({"window": []}, "empty"),
            ({"window": [[1.0, 2.0]]}, "wrong width"),
            ({"window": [[1.0] * 39, [1.0] * 38]}, "ragged"),
            ({}, "missing field"),
            ({"window": "not-a-window"}, "wrong type"),
        ],
    )
    def test_predict_rejects_bad_input(
        self, client: ApiClient, payload: JsonDict, reason: str
    ) -> None:
        assert client.post("/predict", json=payload).status_code == 422, reason

    def test_predict_rejects_non_finite_values(self, client: ApiClient) -> None:
        # Sent as raw text because strict JSON has no Infinity literal, which is
        # exactly how such a payload would arrive from a misbehaving client.
        row = "[" + ", ".join(["Infinity"] + ["0.0"] * (SPEC.n_features - 1)) + "]"
        rows = ", ".join([row] + ["[" + ", ".join(["0.0"] * SPEC.n_features) + "]"] * 49)
        response = client.post(
            "/predict",
            content=f'{{"window": [{rows}]}}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_replay_status_reports_profile_and_seed(self, client: ApiClient) -> None:
        body = client.get("/replay/status").json()
        assert body["fault_profile"] == "clean"
        assert body["seed"] == 1
        assert body["total_frames"] > 0

    def test_correlation_id_is_echoed(self, client: ApiClient) -> None:
        response = client.get("/health", headers={"x-correlation-id": "abc123"})
        assert response.headers["x-correlation-id"] == "abc123"

    def test_metrics_endpoint_is_prometheus_text(self, client: ApiClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "fi_service_ready" in response.text
        assert "# TYPE" in response.text

    def test_unknown_route_is_404(self, client: ApiClient) -> None:
        assert client.get("/does-not-exist").status_code == 404

    def test_model_reports_the_configured_horizon(
        self, client: ApiClient, settings: Settings
    ) -> None:
        """The demo names the horizon in its copy, so it must follow settings."""
        assert client.get("/model").json()["horizon_s"] == settings.window.horizon_s


class TestReplayControl:
    """``POST /replay/control`` is the demo's only write path.

    Restart is folded into the existing command rather than given a route of its
    own, so the published route set asserted in ``test_boundaries`` does not
    change. That makes the *payload combinations* the contract, and the old
    ones must keep working alongside the new field.

    The match here is deliberately tiny. Restarting starts a replay task, and a
    full synthetic match would have it scoring twelve thousand frames in the
    background of an otherwise instant assertion.
    """

    @pytest.fixture
    def client(self, settings: Settings) -> ApiClient:
        short = generate_synthetic_match(seed=5, period_duration_s=4.0)
        metrics = Metrics()
        engine = InsightEngine(
            settings=settings,
            orientation=short.orientation,
            events=short.events,
            frame_rate=short.frame_rate,
            predictor=HeuristicPredictor(),
            metrics=metrics,
        )
        player = ReplayPlayer(
            match_id="synthetic",
            tracking=short.tracking,
            profile=settings.fault_profile("clean"),
            seed=1,
            speed=1.0,
        )
        return ApiClient(TestClient(create_app(settings, engine, player, metrics)))

    def test_pause_only_payload_is_still_accepted(self, client: ApiClient) -> None:
        body = client.post("/replay/control", json={"paused": True}).json()
        assert body["paused"] is True

    def test_speed_only_payload_is_still_accepted(self, client: ApiClient) -> None:
        body = client.post("/replay/control", json={"speed": 5.0}).json()
        assert body["speed"] == 5.0

    def test_restart_rewinds_and_resumes(self, client: ApiClient) -> None:
        client.post("/replay/control", json={"paused": True})
        body = client.post("/replay/control", json={"restart": True}).json()
        assert body["frames_emitted"] == 0
        assert body["match_time_s"] == 0.0
        # A paused loop sits in a sleep and never takes another frame, so it
        # would never apply the engine-side half of the rewind. Restart resumes
        # unless the same command says otherwise.
        assert body["paused"] is False

    def test_restart_can_end_paused_when_asked(self, client: ApiClient) -> None:
        """Restart is applied first, so an explicit `paused` still wins."""
        body = client.post("/replay/control", json={"restart": True, "paused": True}).json()
        assert body["frames_emitted"] == 0
        assert body["paused"] is True

    def test_restart_with_speed_applies_both(self, client: ApiClient) -> None:
        body = client.post("/replay/control", json={"restart": True, "speed": 5.0}).json()
        assert body["frames_emitted"] == 0
        assert body["speed"] == 5.0

    def test_restart_without_a_replay_is_503(self, settings: Settings) -> None:
        bare = ApiClient(TestClient(create_app(settings)))
        assert bare.post("/replay/control", json={"restart": True}).status_code == 503


class TestEpisodeSettingsSensitivity:
    """The grouping knobs move results, so their effect is measured, not assumed."""

    def test_bridge_gap_changes_alarm_count(self) -> None:
        times = np.arange(0, 40, 0.5)
        teams = np.zeros(len(times), dtype=int)
        rng = np.random.default_rng(3)
        probs = rng.random(len(times))
        counts = {
            gap: len(group_alarms(times, teams, probs, 0.6, bridge_gap_s=gap))
            for gap in (0.0, 1.0, 4.0)
        }
        assert counts[0.0] >= counts[1.0] >= counts[4.0]

    def test_defaults_are_explicit(self) -> None:
        settings = EpisodeSettings()
        assert settings.merge_gap_s == 10.0
        assert settings.alarm_bridge_gap_s == 2.0
        assert settings.max_false_alarms_per_90 == 12.0
