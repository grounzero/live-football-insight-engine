"""Building the small ONNX model the public demo serves.

Not the reference model. The evaluated GRU is trained on Metrica tracking data
and reported on a held-out real match; this one is trained on the output of
:mod:`football_insights.data.synthetic` and demonstrates the *path* — features,
sequence model, ONNX export, live scoring — not a measured capability. Its
metadata says so in a field the API returns, so the distinction survives into
anything that reads it.

It exists because the alternative for a hosted demo is worse in both
directions. Shipping the reference checkpoint would mean either publishing a
model trained on data this project is not licensed to redistribute or pulling
one at startup from somewhere; serving the rule-based fallback would mean the
one thing the project is about is the one thing the demo does not do. Training a
small model on data the repository can generate avoids both, and the artifact is
built during the image build rather than committed or trained at startup.

Everything here needs the ``train`` extra. Nothing in the serving path imports
this module.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

import numpy as np

from football_insights.domain import Team
from football_insights.errors import DataValidationError

if TYPE_CHECKING:
    from pathlib import Path

    from football_insights.config import Settings
    from football_insights.models.train import MatchData
    from football_insights.types import JsonDict

#: Name the artifact is registered under. Deliberately not ``gru-temporal``:
#: the two must never be confused in a registry, a log line or a status payload.
DEMO_MODEL_NAME: Final = "demo-synthetic-gru"

#: Fixtures per tactical archetype: one to fit on, one to early-stop and choose
#: a threshold on.
#:
#: Split by whole fixture rather than by window, because a 5 s window at a 0.5 s
#: stride overlaps its neighbours by 90% — splitting windows would put a
#: sample's own neighbours on both sides of the boundary and turn early stopping
#: into a measurement of memorisation. Splitting by *profile* instead would be a
#: different mistake: the model would be validated on an archetype it had never
#: seen, so early stopping would rank generalisation across tactics rather than
#: fit, and the threshold would be chosen against a distribution the model was
#: not trained for.
DEMO_FIXTURES_PER_PROFILE: Final = 2
DEMO_PERIOD_DURATION_S: Final = 600.0

#: Base seed for the development fixtures.
#:
#: Deliberately nowhere near
#: :data:`~football_insights.serving.loader.PUBLIC_FIXTURE_SEED_BASE`, so the
#: six fixtures this model is built from and the three the hosted demo plays
#: cannot overlap. A test asserts the two sets are disjoint rather than trusting
#: the arithmetic to stay that way.
DEMO_SEED_BASE: Final = 20260801

DEMO_NOTES: Final = (
    "Trained only on generated synthetic tracking data, for the hosted demonstration. "
    "This is NOT the reference model: the evaluated GRU is trained and reported on "
    "Metrica sample matches, and its results do not transfer to this artifact. "
    "Use for demonstrating the pipeline, never as a measure of predictive quality."
)


def training_seed(offset: int) -> int:
    """Seed for the development fixture at ``offset``.

    Named and exported so the disjointness from the public rotation's seeds can
    be asserted rather than assumed. The hosted fixtures must never be ones this
    model was fitted, early-stopped or thresholded on — a demo scoring its own
    training data is not a demonstration of anything.
    """
    return DEMO_SEED_BASE + offset


def _prepared_fixtures(settings: Settings, seed: int) -> tuple[list[MatchData], list[MatchData]]:
    """Generate the development fixtures and turn each into a modelling dataset.

    Produces the same :class:`MatchData` that a prepared Metrica match becomes,
    so threshold selection below is the identical code path the reference run
    uses rather than a simplified stand-in.

    Args:
        settings: Resolved configuration; supplies the window and episode knobs.
        seed: Base seed. Each fixture uses a distinct derived seed.

    Returns:
        The training and validation datasets, split by whole fixture with one
        of each per tactical archetype.
    """
    from football_insights.data.pipeline import prepare_parsed_match
    from football_insights.data.synthetic import PROFILES, generate_synthetic_match
    from football_insights.features.window import WindowGeometry
    from football_insights.models.train import MatchData

    train: list[MatchData] = []
    validate: list[MatchData] = []
    for profile_index, profile in enumerate(PROFILES):
        for replicate in range(DEMO_FIXTURES_PER_PROFILE):
            index = profile_index * DEMO_FIXTURES_PER_PROFILE + replicate
            match_id = f"Synthetic_Train_{profile.key}_{replicate}"
            match = generate_synthetic_match(
                seed=seed + index,
                n_periods=2,
                period_duration_s=DEMO_PERIOD_DURATION_S,
                profile=profile,
            )
            prepared = prepare_parsed_match(
                match_id,
                match.tracking,
                match.events,
                match.orientation,
                settings,
            )
            geometry = WindowGeometry.build(settings.window, match.frame_rate)
            labels = prepared.labels
            times = labels.time_s
            data = MatchData(
                match_id=match_id,
                windows=prepared.windows(geometry.observation_frames, geometry.sequence_length),
                labels=labels.label.astype(np.int8),
                times_s=times,
                teams=labels.attacking_team.astype(np.int64),
                cluster_id=labels.cluster_id.astype(np.int64),
                episode_times=np.array([e.entry_time_s for e in labels.episodes], dtype=np.float64),
                episode_teams=np.array([int(e.team == Team.AWAY) for e in labels.episodes]),
                minutes=float(times.max() - times.min()) / 60.0 if times.size else 90.0,
            )
            # The last replicate of each profile is the held-out one, so every
            # archetype is represented on both sides of the split.
            target = validate if replicate == DEMO_FIXTURES_PER_PROFILE - 1 else train
            target.append(data)
    return train, validate


def build_demo_model(settings: Settings, out_dir: Path, *, seed: int = DEMO_SEED_BASE) -> JsonDict:
    """Train a small model on generated data and export it to ONNX.

    Deterministic for a fixed seed: the fixtures come from a seeded generator and
    training goes through ``seed_everything``. Parity between PyTorch and ONNX
    Runtime is checked before anything is written, and a failure raises rather
    than leaving a graph that scores differently from the model it came from.

    Args:
        settings: Resolved configuration.
        out_dir: Registry directory to write the artifact and its metadata into.
        seed: Base seed for fixtures and training.

    Returns:
        A report with the sample counts, positive rate and measured parity.

    Raises:
        DataValidationError: If the fixtures yield too few positives to train on,
            or the exported graph disagrees with the model it came from.
    """
    from football_insights.data.synthetic import PROFILES
    from football_insights.models.evaluate import choose_threshold_by_alarm_budget
    from football_insights.models.export_onnx import check_parity, export
    from football_insights.models.temporal import train_temporal

    train_fixtures, val_fixtures = _prepared_fixtures(settings, seed)
    train_windows = np.concatenate([m.windows for m in train_fixtures])
    train_labels = np.concatenate([m.labels for m in train_fixtures]).astype(np.float32)
    val_windows = np.concatenate([m.windows for m in val_fixtures])
    val_labels = np.concatenate([m.labels for m in val_fixtures]).astype(np.float32)

    positives = int(train_labels.sum())
    if positives < 2 or int(val_labels.sum()) < 1:
        # Early stopping ranks on PR-AUC, which is undefined without positives,
        # so this would otherwise fail deep inside training with a numpy error.
        msg = (
            f"generated fixtures produced too few positive windows to train on "
            f"({positives} in training, {int(val_labels.sum())} in validation). "
            "Increase DEMO_FIXTURES_PER_PROFILE or DEMO_PERIOD_DURATION_S."
        )
        raise DataValidationError(msg)

    tuned = settings.model_copy(deep=True)
    tuned.model.seed = seed
    # Smaller and shorter than the reference run. This model is trained inside a
    # container build, where minutes are the budget, and it is demonstrating the
    # path rather than competing on a metric.
    # Smaller and shorter than the reference run, but not as small as it was.
    # The operating point comes from a false-alarm budget, so the only honest
    # way to make the demo say more is to make the model *better* at the
    # entries it is confident about — lowering the threshold to fill the
    # silence would be manufacturing alarms the budget exists to prevent.
    # Richer motion also gives the features more to separate on, which a
    # 32-unit model trained for 12 epochs could not exploit.
    tuned.model.hidden_size = 48
    tuned.model.max_epochs = 30
    tuned.model.early_stopping_patience = 6

    predictor, history = train_temporal(
        train_windows,
        train_labels,
        val_windows,
        val_labels,
        tuned.model,
        training_matches=tuple(m.match_id for m in train_fixtures),
        dataset_fingerprint=(
            f"synthetic:seed={seed}:profiles={'+'.join(p.key for p in PROFILES)}"
            f":per_profile={DEMO_FIXTURES_PER_PROFILE}"
        ),
        config_fingerprint=tuned.fingerprint(),
    )

    predictor.metadata.metrics.setdefault("epochs_run", float(len(history.train_loss)))

    # An operating point, chosen the way the reference run chooses one: from a
    # false-alarm budget, on training fixtures only. Left at the trained default
    # of 0.5 against an 11% positive rate, the demo would raise an alarm every
    # few seconds and the editorial layer would spend the whole match
    # suppressing it — which demonstrates neither the model nor the editor.
    threshold = choose_threshold_by_alarm_budget(
        per_match=[
            (
                m.times_s,
                m.teams,
                predictor.predict_proba(m.windows),
                m.episode_times,
                m.episode_teams,
                m.minutes,
            )
            for m in train_fixtures
        ],
        horizon_s=settings.window.horizon_s,
        settings=settings.episode,
        max_false_alarms_per_90=settings.episode.max_false_alarms_per_90,
    )

    # `ModelMetadata` is frozen, and the name is the one field that must change:
    # the trainer stamps the reference model's name, and publishing this artifact
    # under it is precisely the confusion this module exists to avoid.
    renamed = replace(
        predictor.metadata, name=DEMO_MODEL_NAME, notes=DEMO_NOTES, threshold=threshold
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export(predictor, out_dir / f"{DEMO_MODEL_NAME}.onnx")
    parity = check_parity(predictor, onnx_path, val_windows[:512])
    if not parity["within_tolerance"]:
        onnx_path.unlink(missing_ok=True)
        msg = (
            f"ONNX export disagrees with the trained model (max absolute difference "
            f"{parity['max_abs_diff']}); refusing to publish a graph that scores differently"
        )
        raise DataValidationError(msg)

    renamed.write(out_dir / f"{DEMO_MODEL_NAME}.metadata.json")
    return {
        "model": DEMO_MODEL_NAME,
        "train_samples": int(train_windows.shape[0]),
        "val_samples": int(val_windows.shape[0]),
        "train_positive_rate": round(float(train_labels.mean()), 4),
        "sequence_length": int(train_windows.shape[1]),
        "n_features": int(train_windows.shape[2]),
        "threshold": renamed.threshold,
        "parity": parity,
        "artifact": str(onnx_path),
    }
