"""Training and evaluation orchestration.

Splits are **match-aware**. With a 0.5 s stride, adjacent windows overlap almost
entirely, so a random split would put near-duplicates of the same attack on both
sides and report a score that has nothing to do with generalisation.

Two things are reported, and both are needed:

* **Leave-one-match-out folds.** Each match is held out in turn. With three
  matches this is the whole of the evidence about generalisation, and the
  fold-to-fold spread is the honest measure of uncertainty.
* **One reference run** on a fixed split, whose artifact is registered and
  served. Its threshold is chosen on the training folds and then frozen.

Confidence intervals from the cluster bootstrap describe within-match sampling
only. They are narrower than the fold spread, and reporting them alone would
overstate what three matches can support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from football_insights.data.pipeline import load_prepared
from football_insights.features.window import WindowGeometry, subsample_indices
from football_insights.models.baseline import BaselinePredictor, train_baseline
from football_insights.models.evaluate import (
    EvaluationResult,
    choose_threshold_by_alarm_budget,
    evaluate,
)
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.models.temporal import TemporalPredictor, train_temporal
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from football_insights.config import Settings
    from football_insights.models.base import Predictor

#: Fraction of the training matches held out, by time, for early stopping.
VALIDATION_TAIL_FRACTION = 0.2

#: Seconds of samples discarded either side of the validation cut. Without it,
#: the last training window and the first validation window overlap and early
#: stopping is judged on data the model has effectively seen.
EMBARGO_S = 30.0


@dataclass(slots=True)
class MatchData:
    """A prepared match loaded into memory."""

    match_id: str
    windows: np.ndarray
    labels: np.ndarray
    times_s: np.ndarray
    teams: np.ndarray
    cluster_id: np.ndarray
    episode_times: np.ndarray
    episode_teams: np.ndarray
    minutes: float

    def __len__(self) -> int:
        """Number of samples."""
        return int(self.labels.shape[0])


@dataclass(slots=True)
class FoldResult:
    """Results for one held-out match."""

    held_out: str
    results: dict[str, EvaluationResult] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialisable form."""
        return {
            "held_out": self.held_out,
            "models": {name: r.to_dict() for name, r in self.results.items()},
        }


def load_matches(settings: Settings, match_ids: list[str] | None = None) -> list[MatchData]:
    """Load prepared matches and materialise their model inputs.

    Args:
        settings: Resolved configuration.
        match_ids: Matches to load; every processed file when omitted.

    Returns:
        The loaded matches, in id order.
    """
    processed = settings.paths.processed_dir
    paths = sorted(processed.glob("*.npz"))
    if match_ids:
        wanted = set(match_ids)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        msg = f"no prepared matches in {processed}; run `football-insights prepare` first"
        raise FileNotFoundError(msg)

    out: list[MatchData] = []
    for path in paths:
        raw = load_prepared(path)
        frame_rate = float(raw["frame_rate"][0])
        geometry = WindowGeometry.build(settings.window, frame_rate)
        picks = subsample_indices(geometry.observation_frames, geometry.sequence_length)

        features = raw["features"]
        end_index = raw["end_index"]
        teams = raw["attacking_team"].astype(np.int64)
        windows = np.empty(
            (end_index.shape[0], geometry.sequence_length, features.shape[2]), dtype=np.float32
        )
        for k, end in enumerate(end_index):
            start = int(end) - geometry.observation_frames + 1
            windows[k] = features[teams[k], start : int(end) + 1][picks]

        times = raw["time_s"]
        out.append(
            MatchData(
                match_id=path.stem,
                windows=windows,
                labels=raw["label"].astype(np.int8),
                times_s=times,
                teams=teams,
                cluster_id=raw["cluster_id"].astype(np.int64),
                episode_times=raw["episode_time"],
                episode_teams=raw["episode_team"].astype(np.int64),
                minutes=float(times.max() - times.min()) / 60.0 if times.size else 90.0,
            )
        )
    return out


def _concatenate(matches: list[MatchData]) -> tuple[np.ndarray, np.ndarray]:
    """Stack windows and labels across matches."""
    return (
        np.concatenate([m.windows for m in matches]),
        np.concatenate([m.labels for m in matches]),
    )


def time_split_with_embargo(
    matches: list[MatchData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split training matches into train and validation by time, with an embargo.

    Each training match contributes its final fraction to validation. The gap
    either side of the cut is discarded, because windows spanning it overlap and
    would make early stopping judge the model on data it has already seen.

    Args:
        matches: Training matches.

    Returns:
        ``(train windows, train labels, validation windows, validation labels)``.
    """
    train_w, train_y, val_w, val_y = [], [], [], []
    for match in matches:
        if len(match) == 0:
            continue
        cut = float(np.quantile(match.times_s, 1.0 - VALIDATION_TAIL_FRACTION))
        train_mask = match.times_s < cut - EMBARGO_S
        val_mask = match.times_s >= cut + EMBARGO_S
        train_w.append(match.windows[train_mask])
        train_y.append(match.labels[train_mask])
        val_w.append(match.windows[val_mask])
        val_y.append(match.labels[val_mask])
    return (
        np.concatenate(train_w),
        np.concatenate(train_y),
        np.concatenate(val_w),
        np.concatenate(val_y),
    )


def build_predictors(
    train_matches: list[MatchData],
    settings: Settings,
    dataset_fingerprint: str | None,
) -> dict[str, Predictor]:
    """Train every model on the same training data.

    Args:
        train_matches: Matches to train on.
        settings: Resolved configuration.
        dataset_fingerprint: Recorded in each model's metadata.

    Returns:
        Predictors keyed by name, including the untrained fallback so it is
        measured on the same footing.
    """
    windows, labels = _concatenate(train_matches)
    ids = tuple(m.match_id for m in train_matches)
    config = settings.fingerprint()

    predictors: dict[str, Predictor] = {
        "heuristic-fallback": HeuristicPredictor(settings.model.threshold)
    }
    for kind in ("logistic", "gbdt"):
        predictors[f"baseline-{kind}"] = train_baseline(
            kind,
            windows,
            labels,
            seed=settings.model.seed,
            training_matches=ids,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config,
        )

    train_w, train_y, val_w, val_y = time_split_with_embargo(train_matches)
    model, history = train_temporal(
        train_w,
        train_y,
        val_w,
        val_y,
        settings.model,
        training_matches=ids,
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config,
    )
    predictors["gru-temporal"] = model
    predictors["gru-temporal"].metadata.metrics.setdefault(
        "epochs_run", float(len(history.train_loss))
    )
    return predictors


def _select_thresholds(
    predictors: dict[str, Predictor],
    train_matches: list[MatchData],
    settings: Settings,
) -> dict[str, float]:
    """Choose each model's operating point on the training matches only.

    Held-out data never influences the threshold, so the reported precision and
    recall are not the product of tuning against the thing being measured.
    """
    thresholds: dict[str, float] = {}
    for name, predictor in predictors.items():
        per_match = [
            (
                m.times_s,
                m.teams,
                predictor.predict_proba(m.windows),
                m.episode_times,
                m.episode_teams,
                m.minutes,
            )
            for m in train_matches
        ]
        thresholds[name] = choose_threshold_by_alarm_budget(
            per_match=per_match,
            horizon_s=settings.window.horizon_s,
            settings=settings.episode,
            max_false_alarms_per_90=settings.episode.max_false_alarms_per_90,
        )
    return thresholds


def evaluate_fold(
    predictors: dict[str, Predictor],
    held_out: MatchData,
    thresholds: dict[str, float],
    settings: Settings,
    bootstrap: bool = True,
) -> FoldResult:
    """Evaluate every predictor on one held-out match."""
    fold = FoldResult(held_out=held_out.match_id)
    for name, predictor in predictors.items():
        probabilities = predictor.predict_proba(held_out.windows)
        fold.results[name] = evaluate(
            name=name,
            y_true=held_out.labels,
            y_prob=probabilities,
            times_s=held_out.times_s,
            teams=held_out.teams,
            cluster_id=held_out.cluster_id,
            episode_times=held_out.episode_times,
            episode_teams=held_out.episode_teams,
            threshold=thresholds[name],
            horizon_s=settings.window.horizon_s,
            settings=settings.episode,
            minutes=held_out.minutes,
            bootstrap=bootstrap,
        )
    return fold


def run_cross_validation(
    settings: Settings,
    match_ids: list[str] | None = None,
    dataset_fingerprint: str | None = None,
    bootstrap: bool = True,
) -> JsonDict:
    """Leave-one-match-out evaluation across every prepared match.

    Args:
        settings: Resolved configuration.
        match_ids: Matches to include.
        dataset_fingerprint: Recorded in model metadata.
        bootstrap: Whether to compute confidence intervals.

    Returns:
        A report with per-fold results and their aggregate.
    """
    matches = load_matches(settings, match_ids)
    if len(matches) < 2:
        msg = "leave-one-match-out needs at least two prepared matches"
        raise ValueError(msg)

    folds: list[FoldResult] = []
    for index, held_out in enumerate(matches):
        train_matches = [m for j, m in enumerate(matches) if j != index]
        predictors = build_predictors(train_matches, settings, dataset_fingerprint)

        thresholds = _select_thresholds(predictors, train_matches, settings)
        folds.append(evaluate_fold(predictors, held_out, thresholds, settings, bootstrap))

    names = sorted(folds[0].results)
    aggregate: JsonDict = {}
    for name in names:
        episode_precision = [f.results[name].episode.precision for f in folds]
        episode_recall = [f.results[name].episode.recall for f in folds]
        pr_auc = [f.results[name].window.pr_auc for f in folds]
        aggregate[name] = {
            "folds": [f.held_out for f in folds],
            "window_pr_auc": [round(v, 4) for v in pr_auc],
            "window_pr_auc_mean": round(float(np.nanmean(pr_auc)), 4),
            "episode_precision": [round(v, 4) for v in episode_precision],
            "episode_recall": [round(v, 4) for v in episode_recall],
            "episode_precision_mean": round(float(np.mean(episode_precision)), 4),
            "episode_recall_mean": round(float(np.mean(episode_recall)), 4),
            "false_alarms_per_90": [
                round(f.results[name].episode.false_alarms_per_90, 2) for f in folds
            ],
            "median_warning_s": [round(f.results[name].episode.median_warning_s, 2) for f in folds],
        }

    return {
        "protocol": "leave-one-match-out",
        "matches": [m.match_id for m in matches],
        "n_samples": int(sum(len(m) for m in matches)),
        "n_episodes": int(sum(len(m.episode_times) for m in matches)),
        "config_fingerprint": settings.fingerprint(),
        "dataset_fingerprint": dataset_fingerprint,
        "caveat": (
            "Three matches. Fold-to-fold spread is the honest measure of "
            "generalisation; bootstrap intervals cover within-match sampling only."
        ),
        "aggregate": aggregate,
        "folds": [f.to_dict() for f in folds],
    }


def train_reference(
    settings: Settings,
    test_match: str,
    match_ids: list[str] | None = None,
    dataset_fingerprint: str | None = None,
) -> JsonDict:
    """Train the reference models on a fixed split and register the artifacts.

    Args:
        settings: Resolved configuration.
        test_match: Match held out for reporting.
        match_ids: Matches to include.
        dataset_fingerprint: Recorded in model metadata.

    Returns:
        A report describing the run.
    """
    matches = load_matches(settings, match_ids)
    held_out = next((m for m in matches if m.match_id == test_match), None)
    if held_out is None:
        available = ", ".join(m.match_id for m in matches)
        msg = f"test match {test_match!r} is not prepared; available: {available}"
        raise ValueError(msg)
    train_matches = [m for m in matches if m.match_id != test_match]

    predictors = build_predictors(train_matches, settings, dataset_fingerprint)
    thresholds = _select_thresholds(predictors, train_matches, settings)
    fold = evaluate_fold(predictors, held_out, thresholds, settings)

    registry = settings.paths.registry_dir
    registry.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for name, predictor in predictors.items():
        metadata = predictor.metadata
        object.__setattr__(metadata, "threshold", thresholds[name])
        for key, value in fold.results[name].window.to_dict().items():
            metadata.metrics[f"test_window_{key}"] = float(value)
        metadata.metrics["test_episode_precision"] = fold.results[name].episode.precision
        metadata.metrics["test_episode_recall"] = fold.results[name].episode.recall

        if isinstance(predictor, BaselinePredictor):
            path = registry / f"{name}.pkl"
            predictor.save(path)
        elif isinstance(predictor, TemporalPredictor):
            path = registry / f"{name}.pt"
            predictor.save(path)
        else:
            path = registry / f"{name}.json"
        metadata.write(registry / f"{name}.metadata.json")
        artifacts[name] = str(path)

    report = {
        "protocol": "fixed reference split",
        "train_matches": [m.match_id for m in train_matches],
        "test_match": test_match,
        "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "artifacts": artifacts,
        "config_fingerprint": settings.fingerprint(),
        "dataset_fingerprint": dataset_fingerprint,
        "results": fold.to_dict(),
    }
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.paths.reports_dir / "reference_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    return report


def write_cross_validation_report(report: JsonDict, path: Path) -> None:
    """Write the cross-validation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
