"""Data-quality and drift monitoring.

Scoped to what is genuinely built. This produces a report and flags threshold
breaches; it does not retrain anything, and nothing in the project claims it
does. For a system fed by one provider's tracking, the failures worth catching
are mundane: a feed whose coordinate range has shifted, a match with far more
missing frames than usual, a model whose confidence distribution has moved.

Comparisons are made between two prepared matches — one reference, one target —
which is the shape the sample dataset allows. The same checks would run against
a rolling production baseline unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from football_insights.data.pipeline import load_prepared
from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from pathlib import Path

    from football_insights.config import Settings

#: Population Stability Index above which a feature is flagged. 0.25 is the
#: conventional "material shift" line; 0.1 to 0.25 is a moderate shift.
PSI_THRESHOLD = 0.25

#: Absolute change in missing-value rate treated as a breach.
MISSING_RATE_DELTA = 0.15

#: Relative change in positive-event rate treated as a breach.
EVENT_RATE_RELATIVE_DELTA = 0.5


@dataclass(frozen=True, slots=True)
class Check:
    """One monitoring check and its result."""

    name: str
    category: str
    value: float
    threshold: float
    violation: bool
    detail: str

    def to_dict(self) -> JsonDict:
        """Serialisable form."""
        payload = asdict(self)
        payload["value"] = round(self.value, 5)
        return payload


def population_stability_index(reference: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature.

    Bin edges come from the reference quantiles, so the measure is insensitive
    to the feature's units and shape. A small floor prevents an empty target bin
    from producing an infinite score.

    Args:
        reference: Reference sample.
        target: Target sample.
        bins: Number of quantile bins.

    Returns:
        The PSI; 0 means identical distributions.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    reference_hist, _ = np.histogram(reference, bins=edges)
    target_hist, _ = np.histogram(target, bins=edges)
    floor = 1e-6
    p = np.maximum(reference_hist / max(reference_hist.sum(), 1), floor)
    q = np.maximum(target_hist / max(target_hist.sum(), 1), floor)
    return float(np.sum((q - p) * np.log(q / p)))


def schema_checks(reference: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> list[Check]:
    """Verify the two datasets share a schema."""
    checks: list[Check] = []
    ref_features = reference["features"].shape[-1]
    tgt_features = target["features"].shape[-1]
    checks.append(
        Check(
            name="feature_count",
            category="schema",
            value=float(tgt_features),
            threshold=float(ref_features),
            violation=ref_features != tgt_features,
            detail=f"reference has {ref_features} features, target has {tgt_features}",
        )
    )
    missing_keys = set(reference) - set(target)
    checks.append(
        Check(
            name="array_keys",
            category="schema",
            value=float(len(missing_keys)),
            threshold=0.0,
            violation=bool(missing_keys),
            detail=f"arrays present in reference but not target: {sorted(missing_keys) or 'none'}",
        )
    )
    return checks


def quality_checks(reference: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> list[Check]:
    """Compare data-quality characteristics."""
    checks: list[Check] = []
    ref_missing = float(np.mean(~np.isfinite(reference["features"])))
    tgt_missing = float(np.mean(~np.isfinite(target["features"])))
    delta = abs(tgt_missing - ref_missing)
    checks.append(
        Check(
            name="non_finite_feature_rate",
            category="quality",
            value=delta,
            threshold=MISSING_RATE_DELTA,
            violation=delta > MISSING_RATE_DELTA,
            detail=f"reference {ref_missing:.4%}, target {tgt_missing:.4%}",
        )
    )

    ref_rate = float(reference["label"].mean()) if reference["label"].size else 0.0
    tgt_rate = float(target["label"].mean()) if target["label"].size else 0.0
    relative = abs(tgt_rate - ref_rate) / max(ref_rate, 1e-9)
    checks.append(
        Check(
            name="positive_event_rate",
            category="event_rate",
            value=relative,
            threshold=EVENT_RATE_RELATIVE_DELTA,
            violation=relative > EVENT_RATE_RELATIVE_DELTA,
            detail=f"reference {ref_rate:.3%}, target {tgt_rate:.3%}",
        )
    )
    return checks


def range_checks(
    target: dict[str, np.ndarray], spec: FeatureSpec = DEFAULT_FEATURE_SPEC
) -> list[Check]:
    """Check normalised coordinate features stay inside their expected range.

    ``ball_x`` is normalised to ``[-1, 1]`` by construction, so values outside it
    mean the coordinate convention or pitch dimensions have changed — the kind of
    fault that silently corrupts every downstream feature.
    """
    features = target["features"]
    index = spec.index("ball_x")
    values = features[..., index]
    outside = float(np.mean(np.abs(values) > 1.05))
    return [
        Check(
            name="ball_x_in_range",
            category="range",
            value=outside,
            threshold=0.01,
            violation=outside > 0.01,
            detail=f"{outside:.4%} of ball_x values outside [-1.05, 1.05]",
        )
    ]


def feature_drift_checks(
    reference: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    watch: tuple[str, ...] = (
        "ball_x",
        "ball_dist_to_goal",
        "nearest_defender_dist",
        "defensive_line_x",
        "attackers_ahead_of_ball",
        "possession_duration_s",
    ),
) -> list[Check]:
    """Compare selected feature distributions.

    A subset rather than all 39: these are the features the model leans on most
    and the ones whose drift would be interpretable to someone on call.
    """
    checks: list[Check] = []
    for name in watch:
        index = spec.index(name)
        ref = reference["features"][..., index].ravel()
        tgt = target["features"][..., index].ravel()
        psi = population_stability_index(ref, tgt)
        checks.append(
            Check(
                name=f"psi_{name}",
                category="feature_drift",
                value=psi,
                threshold=PSI_THRESHOLD,
                violation=psi > PSI_THRESHOLD,
                detail=(
                    f"PSI {psi:.3f} (reference mean {ref.mean():.2f}, target mean {tgt.mean():.2f})"
                ),
            )
        )
    return checks


def confidence_drift_check(reference_scores: np.ndarray, target_scores: np.ndarray) -> Check:
    """Compare two confidence distributions.

    A model whose confidence distribution has shifted is worth investigating
    even when accuracy looks unchanged: it usually means the input distribution
    moved, and accuracy will follow.
    """
    psi = population_stability_index(reference_scores, target_scores)
    return Check(
        name="psi_model_confidence",
        category="confidence_drift",
        value=psi,
        threshold=PSI_THRESHOLD,
        violation=psi > PSI_THRESHOLD,
        detail=(
            f"PSI {psi:.3f} (reference mean {reference_scores.mean():.3f}, "
            f"target mean {target_scores.mean():.3f})"
        ),
    )


def drift_report(
    settings: Settings,
    reference_match: str,
    target_match: str,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> JsonDict:
    """Run every check between two prepared matches.

    Args:
        settings: Resolved configuration.
        reference_match: Baseline match id.
        target_match: Match to compare against the baseline.
        spec: Feature schema.

    Returns:
        The report, listing every check and how many were violated.
    """
    processed = settings.paths.processed_dir
    reference = load_prepared(processed / f"{reference_match}.npz")
    target = load_prepared(processed / f"{target_match}.npz")

    checks = [
        *schema_checks(reference, target),
        *quality_checks(reference, target),
        *range_checks(target, spec),
        *feature_drift_checks(reference, target, spec),
    ]
    violations = sum(1 for c in checks if c.violation)
    return {
        "reference_match": reference_match,
        "target_match": target_match,
        "feature_schema": spec.schema_hash,
        "checks_run": len(checks),
        "violations": violations,
        "psi_threshold": PSI_THRESHOLD,
        "note": (
            "Reports and flags only. No automated retraining is implemented, and none is claimed."
        ),
        "details": [c.to_dict() for c in checks],
    }


def write_drift_report(report: JsonDict, path: Path) -> None:
    """Write a drift report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
