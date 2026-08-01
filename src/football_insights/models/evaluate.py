"""Evaluation: window-level, episode-level and cluster bootstrap.

Window-level metrics are reported because they are standard, but they are not
the headline. With a 0.5 s stride and a 5 s horizon a single penalty-area entry
generates about ten near-identical positive windows, so window counts measure
how long situations last as much as how well the model detects them.

Episode-level metrics answer the question a live product actually asks: of the
attacking passages that reached the box, how many did the system flag, how far
ahead, and how often did it cry wolf.

Confidence intervals use a **cluster bootstrap over possession sequences**.
Resampling individual windows would treat ten views of one attack as ten
independent observations and produce intervals far tighter than the evidence
supports. With only three matches, between-match variance cannot be captured by
any interval, so fold-wise point estimates are always reported alongside.
"""

from __future__ import annotations

import warnings as warnings_module
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from football_insights.types import JsonDict

if TYPE_CHECKING:
    from football_insights.config import EpisodeSettings

#: Minutes in a match, for normalising false-alarm rates.
MATCH_MINUTES = 90.0


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    """Standard classification metrics over independent-looking windows."""

    n: int
    positives: int
    precision: float
    recall: float
    f1: float
    pr_auc: float
    brier: float
    base_rate: float

    def to_dict(self) -> dict[str, float | int]:
        """Serialisable form."""
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Metrics over merged attacking episodes and grouped alarms."""

    episodes: int
    alarms: int
    detected: int
    false_alarms: int
    early_alarms: int
    precision: float
    recall: float
    f1: float
    false_alarms_per_90: float
    median_warning_s: float
    warning_iqr_s: tuple[float, float]
    median_windows_per_alarm: float

    def to_dict(self) -> JsonDict:
        """Serialisable form."""
        payload = asdict(self)
        payload["warning_iqr_s"] = [round(v, 3) for v in self.warning_iqr_s]
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in payload.items()}


@dataclass(frozen=True, slots=True)
class Alarm:
    """A maximal run of consecutive above-threshold windows."""

    start_s: float
    end_s: float
    team: int
    n_windows: int
    peak: float


@dataclass(slots=True)
class EvaluationResult:
    """Everything computed for one model on one held-out set."""

    name: str
    threshold: float
    window: WindowMetrics
    episode: EpisodeMetrics
    calibration: list[dict[str, float]] = field(default_factory=list)
    intervals: dict[str, list[float]] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> JsonDict:
        """Serialisable form for the evaluation report."""
        return {
            "name": self.name,
            "threshold": round(self.threshold, 4),
            "window": self.window.to_dict(),
            "episode": self.episode.to_dict(),
            "calibration": self.calibration,
            "confidence_intervals_95": self.intervals,
            "notes": self.notes,
        }


def window_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> WindowMetrics:
    """Compute window-level metrics.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        The metrics. Precision is defined as 0 when nothing fires, which is the
        honest reading: a model that never predicts positive has no precision to
        report, and treating it as 1.0 would flatter it.
    """
    predicted = y_prob >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    positives = int(y_true.sum())
    pr_auc = (
        float(average_precision_score(y_true, y_prob))
        if 0 < positives < len(y_true)
        else float("nan")
    )
    return WindowMetrics(
        n=len(y_true),
        positives=positives,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        brier=float(brier_score_loss(y_true, y_prob)) if len(y_true) else float("nan"),
        base_rate=float(y_true.mean()) if len(y_true) else 0.0,
    )


def group_alarms(
    times_s: np.ndarray,
    teams: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    bridge_gap_s: float,
) -> list[Alarm]:
    """Group consecutive firing windows into alarms.

    Bridging short gaps absorbs probability flicker around the threshold: a
    model that dips below for a single window has not stopped and restarted
    warning, and counting that as two alarms would double its apparent
    false-alarm rate.

    Args:
        times_s: Prediction time per window, ascending.
        teams: Attacking team code per window.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        bridge_gap_s: Gaps up to this length are bridged.

    Returns:
        Alarms in time order.
    """
    firing = np.flatnonzero(y_prob >= threshold)
    alarms: list[Alarm] = []
    if firing.size == 0:
        return alarms

    start = firing[0]
    previous = firing[0]
    for index in firing[1:]:
        contiguous = (times_s[index] - times_s[previous] <= bridge_gap_s) and (
            teams[index] == teams[previous]
        )
        if not contiguous:
            alarms.append(_make_alarm(times_s, teams, y_prob, start, previous))
            start = index
        previous = index
    alarms.append(_make_alarm(times_s, teams, y_prob, start, previous))
    return alarms


def _make_alarm(
    times_s: np.ndarray, teams: np.ndarray, y_prob: np.ndarray, start: int, end: int
) -> Alarm:
    """Build one alarm from an index range."""
    return Alarm(
        start_s=float(times_s[start]),
        end_s=float(times_s[end]),
        team=int(teams[start]),
        n_windows=int(end - start + 1),
        peak=float(y_prob[start : end + 1].max()),
    )


def episode_metrics(
    *,
    times_s: np.ndarray,
    teams: np.ndarray,
    y_prob: np.ndarray,
    episode_times: np.ndarray,
    episode_teams: np.ndarray,
    threshold: float,
    horizon_s: float,
    bridge_gap_s: float,
    minutes: float = MATCH_MINUTES,
) -> EpisodeMetrics:
    """Score grouped alarms against merged ground-truth episodes.

    An alarm counts as a detection when it overlaps the lead-up interval
    ``[entry - horizon, entry]`` of an unmatched episode by the same team.
    Matching is greedy by earliest alarm, and each alarm and episode is used at
    most once.

    Alarms that fire *before* the lead-up interval are counted separately as
    "early" rather than silently scored as false alarms. They fall outside the
    model's stated contract, but a warning twelve seconds ahead of an entry is
    not the same kind of error as one during a goal kick, and burying them in
    the false-alarm count would understate the system.

    Args:
        times_s: Prediction time per window.
        teams: Attacking team code per window.
        y_prob: Predicted probabilities.
        episode_times: Ground-truth entry times.
        episode_teams: Team code per episode.
        threshold: Decision threshold.
        horizon_s: Prediction horizon.
        bridge_gap_s: Alarm bridging tolerance.
        minutes: Match length used to normalise the false-alarm rate.

    Returns:
        Episode-level metrics.
    """
    alarms = group_alarms(times_s, teams, y_prob, threshold, bridge_gap_s)
    matched_episode: dict[int, int] = {}
    matched_alarm: set[int] = set()
    warnings: list[float] = []
    early = 0

    for a_index, alarm in enumerate(alarms):
        best: int | None = None
        for e_index in range(len(episode_times)):
            if e_index in matched_episode.values() or episode_teams[e_index] != alarm.team:
                continue
            entry = float(episode_times[e_index])
            lead_start, lead_end = entry - horizon_s, entry
            if alarm.start_s <= lead_end and alarm.end_s >= lead_start:
                best = e_index
                break
        if best is not None:
            matched_episode[a_index] = best
            matched_alarm.add(a_index)
            warnings.append(float(episode_times[best]) - alarm.start_s)

    unmatched = [a for i, a in enumerate(alarms) if i not in matched_alarm]
    for alarm in unmatched:
        same_team = episode_times[episode_teams == alarm.team]
        if same_team.size and np.any(
            (same_team > alarm.end_s) & (same_team - alarm.end_s <= 3 * horizon_s)
        ):
            early += 1

    detected = len(matched_alarm)
    false_alarms = len(alarms) - detected
    n_episodes = len(episode_times)
    precision = detected / len(alarms) if alarms else 0.0
    recall = detected / n_episodes if n_episodes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    warn = np.array(warnings) if warnings else np.array([np.nan])
    per_alarm = [a.n_windows for a in alarms] or [0]

    # No matched episode means no warning time to report; NaN is the honest
    # answer and the all-NaN slice warning is expected rather than a defect.
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore", RuntimeWarning)
        median_warning = float(np.nanmedian(warn))
        warning_iqr = (
            float(np.nanpercentile(warn, 25)),
            float(np.nanpercentile(warn, 75)),
        )

    return EpisodeMetrics(
        episodes=n_episodes,
        alarms=len(alarms),
        detected=detected,
        false_alarms=false_alarms,
        early_alarms=early,
        precision=precision,
        recall=recall,
        f1=f1,
        false_alarms_per_90=false_alarms / max(minutes, 1e-9) * MATCH_MINUTES,
        median_warning_s=median_warning,
        warning_iqr_s=warning_iqr,
        median_windows_per_alarm=float(np.median(per_alarm)),
    )


def calibration_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Reliability curve as equal-width probability bins.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        One record per non-empty bin with predicted and observed frequencies.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for lo, hi in pairwise(edges):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        out.append(
            {
                "bin_lower": round(float(lo), 3),
                "bin_upper": round(float(hi), 3),
                "count": int(mask.sum()),
                "mean_predicted": round(float(y_prob[mask].mean()), 5),
                "observed_rate": round(float(y_true[mask].mean()), 5),
            }
        )
    return out


def cluster_bootstrap(
    *,
    cluster_id: np.ndarray,
    times_s: np.ndarray,
    teams: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    episode_times: np.ndarray,
    episode_teams: np.ndarray,
    threshold: float,
    horizon_s: float,
    settings: EpisodeSettings,
    minutes: float = MATCH_MINUTES,
) -> dict[str, list[float]]:
    """Confidence intervals by resampling whole possession sequences.

    The resampling unit is the possession cluster, never the individual window.
    Each replicate draws clusters with replacement, keeps every window belonging
    to a drawn cluster, and recomputes the metrics; ground-truth episodes are
    carried along with the clusters that contain them.

    Args:
        cluster_id: Possession cluster per window.
        times_s: Prediction time per window.
        teams: Attacking team code per window.
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        episode_times: Ground-truth entry times.
        episode_teams: Team code per episode.
        threshold: Decision threshold.
        horizon_s: Prediction horizon.
        settings: Bootstrap configuration.
        minutes: Match length for the false-alarm rate.

    Returns:
        Percentile intervals keyed by metric name.
    """
    rng = np.random.default_rng(settings.bootstrap_seed)
    clusters = np.unique(cluster_id)
    if clusters.size < 2:
        return {}

    # Attribute each episode to the cluster whose windows bracket it, so an
    # episode is resampled with the passage of play that produced it.
    episode_cluster = np.full(episode_times.shape[0], -1, dtype=np.int64)
    for e_index, entry in enumerate(episode_times):
        candidates = np.flatnonzero(
            (times_s <= entry) & (times_s >= entry - horizon_s) & (teams == episode_teams[e_index])
        )
        if candidates.size:
            episode_cluster[e_index] = cluster_id[candidates[-1]]

    collected: dict[str, list[float]] = {
        "window_precision": [],
        "window_recall": [],
        "window_pr_auc": [],
        "episode_precision": [],
        "episode_recall": [],
        "false_alarms_per_90": [],
    }
    index_by_cluster = {c: np.flatnonzero(cluster_id == c) for c in clusters}

    for _ in range(settings.bootstrap_replicates):
        drawn = rng.choice(clusters, size=clusters.size, replace=True)
        rows = np.concatenate([index_by_cluster[c] for c in drawn])
        if rows.size == 0:
            continue
        order = np.argsort(times_s[rows], kind="stable")
        rows = rows[order]

        counts: dict[int, int] = {}
        for c in drawn:
            counts[int(c)] = counts.get(int(c), 0) + 1
        keep_episodes = np.concatenate(
            [
                np.flatnonzero(episode_cluster == c).repeat(n)
                for c, n in counts.items()
                if np.any(episode_cluster == c)
            ]
            or [np.array([], dtype=np.int64)]
        )

        wm = window_metrics(y_true[rows], y_prob[rows], threshold)
        em = episode_metrics(
            times_s=times_s[rows],
            teams=teams[rows],
            y_prob=y_prob[rows],
            episode_times=episode_times[keep_episodes],
            episode_teams=episode_teams[keep_episodes],
            threshold=threshold,
            horizon_s=horizon_s,
            bridge_gap_s=settings.alarm_bridge_gap_s,
            minutes=minutes,
        )
        collected["window_precision"].append(wm.precision)
        collected["window_recall"].append(wm.recall)
        if np.isfinite(wm.pr_auc):
            collected["window_pr_auc"].append(wm.pr_auc)
        collected["episode_precision"].append(em.precision)
        collected["episode_recall"].append(em.recall)
        collected["false_alarms_per_90"].append(em.false_alarms_per_90)

    alpha = (1.0 - settings.confidence_level) / 2.0
    return {
        name: [
            round(float(np.percentile(values, 100 * alpha)), 5),
            round(float(np.percentile(values, 100 * (1 - alpha))), 5),
        ]
        for name, values in collected.items()
        if values
    }


def evaluate(
    *,
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    times_s: np.ndarray,
    teams: np.ndarray,
    cluster_id: np.ndarray,
    episode_times: np.ndarray,
    episode_teams: np.ndarray,
    threshold: float,
    horizon_s: float,
    settings: EpisodeSettings,
    minutes: float = MATCH_MINUTES,
    bootstrap: bool = True,
) -> EvaluationResult:
    """Run the full evaluation for one model on one held-out set.

    Args:
        name: Model name for the report.
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        times_s: Prediction time per window.
        teams: Attacking team code per window.
        cluster_id: Possession cluster per window.
        episode_times: Ground-truth entry times.
        episode_teams: Team code per episode.
        threshold: Decision threshold.
        horizon_s: Prediction horizon.
        settings: Episode and bootstrap configuration.
        minutes: Match length for rate normalisation.
        bootstrap: Whether to compute confidence intervals.

    Returns:
        The evaluation result.
    """
    return EvaluationResult(
        name=name,
        threshold=threshold,
        window=window_metrics(y_true, y_prob, threshold),
        episode=episode_metrics(
            times_s=times_s,
            teams=teams,
            y_prob=y_prob,
            episode_times=episode_times,
            episode_teams=episode_teams,
            threshold=threshold,
            horizon_s=horizon_s,
            bridge_gap_s=settings.alarm_bridge_gap_s,
            minutes=minutes,
        ),
        calibration=calibration_bins(y_true, y_prob),
        intervals=(
            cluster_bootstrap(
                cluster_id=cluster_id,
                times_s=times_s,
                teams=teams,
                y_true=y_true,
                y_prob=y_prob,
                episode_times=episode_times,
                episode_teams=episode_teams,
                threshold=threshold,
                horizon_s=horizon_s,
                settings=settings,
                minutes=minutes,
            )
            if bootstrap
            else {}
        ),
    )


def choose_threshold_by_alarm_budget(
    *,
    per_match: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    horizon_s: float,
    settings: EpisodeSettings,
    max_false_alarms_per_90: float = 12.0,
    minimum: float = 0.05,
) -> float:
    """Pick a threshold from a false-alarm budget rather than window precision.

    Choosing by window precision optimises the wrong thing for this product.
    Windows are not what a viewer experiences: a threshold that looks acceptable
    per window can still raise an alarm every forty seconds, which no broadcast
    would tolerate. Measured on the training matches, targeting window precision
    of 0.30 produced roughly 140 false alarms per 90 minutes.

    So the operating point is set by the constraint that actually binds — how
    often the system may interrupt — and recall is whatever that budget affords.
    The threshold is still chosen on training matches only and then frozen.

    Args:
        per_match: One tuple per training match of ``(times, teams, probabilities,
            episode times, episode teams, minutes)``.
        horizon_s: Prediction horizon.
        settings: Episode grouping configuration.
        max_false_alarms_per_90: Budget for unmatched alarms per 90 minutes.
        minimum: Lowest threshold to consider.

    Returns:
        The lowest threshold meeting the budget, maximising recall subject to
        it. Falls back to the threshold with fewest false alarms if the budget
        cannot be met anywhere.
    """
    grid = np.linspace(minimum, 0.98, 94)
    best_threshold = float(grid[-1])
    best_rate = np.inf
    for threshold in grid:
        false_alarms = 0.0
        minutes = 0.0
        detected = 0
        episodes = 0
        for times, teams, probabilities, ep_times, ep_teams, match_minutes in per_match:
            metrics = episode_metrics(
                times_s=times,
                teams=teams,
                y_prob=probabilities,
                episode_times=ep_times,
                episode_teams=ep_teams,
                threshold=float(threshold),
                horizon_s=horizon_s,
                bridge_gap_s=settings.alarm_bridge_gap_s,
                minutes=match_minutes,
            )
            false_alarms += metrics.false_alarms
            minutes += match_minutes
            detected += metrics.detected
            episodes += metrics.episodes
        rate = false_alarms / max(minutes, 1e-9) * MATCH_MINUTES
        if rate < best_rate:
            best_rate, best_threshold = rate, float(threshold)
        if rate <= max_false_alarms_per_90 and detected > 0:
            return float(threshold)
    return best_threshold


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.30,
    minimum: float = 0.05,
) -> float:
    """Pick a decision threshold on *training* data.

    Selected as the lowest threshold reaching a target precision, which
    maximises recall subject to a false-alarm budget a viewer-facing product can
    tolerate. Chosen on training folds only and then frozen: tuning it on the
    held-out match would make every reported number optimistic.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        target_precision: Precision to reach if possible.
        minimum: Lowest threshold to consider.

    Returns:
        The chosen threshold, falling back to the best-F1 threshold when the
        target precision is unreachable.
    """
    grid = np.linspace(minimum, 0.95, 91)
    best_f1, best_f1_threshold = -1.0, 0.5
    for threshold in grid:
        metrics = window_metrics(y_true, y_prob, float(threshold))
        if metrics.precision >= target_precision and metrics.recall > 0:
            return float(threshold)
        if metrics.f1 > best_f1:
            best_f1, best_f1_threshold = metrics.f1, float(threshold)
    return best_f1_threshold
