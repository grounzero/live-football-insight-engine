"""Deterministic insight wording.

No language model is involved. Templates are fixed strings selected by
probability band and context, so the same window always produces the same
sentence — a property the determinism tests rely on and a prerequisite for
editorial review of a live product.

Every headline is hedged. The model estimates a probability over a short
horizon; presenting that as a statement of fact ("they *will* get in behind")
would misrepresent it to a viewer. :func:`is_hedged` is applied in tests to
every emitted headline so an unhedged template cannot slip in.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.insight.types import ContextFact, InsightKind

#: Words and phrases that mark a statement as an estimate rather than a fact.
#: An emitted headline must contain at least one.
HEDGE_TERMS: Final[frozenset[str]] = frozenset(
    {
        "building",
        "increasing",
        "elevated",
        "growing",
        "suggests",
        "indicates",
        "looks",
        "appears",
        "signs of",
        "threatening",
        "starting to",
        "beginning to",
        "chance of",
        "likely",
        "may",
        "could",
    }
)

#: Phrases that would assert a future event as fact. Tests reject any template
#: containing one.
FORBIDDEN_ASSERTIONS: Final[frozenset[str]] = frozenset(
    {"will score", "will enter", "is going to", "certain", "guaranteed", "definitely"}
)

_HEADLINES: Final[dict[InsightKind, str]] = {
    InsightKind.BUILDING_THREAT: "Attacking threat is building",
    InsightKind.ELEVATED_ENTRY_CHANCE: (
        "Movement suggests an elevated chance of a penalty-area entry"
    ),
    InsightKind.SUSTAINED_PRESSURE: "Signs of sustained pressure building",
}


def is_hedged(text: str) -> bool:
    """Whether a headline is appropriately qualified.

    Args:
        text: Candidate headline.

    Returns:
        ``True`` if it contains a hedge term and no forbidden assertion.
    """
    lowered = text.lower()
    if any(bad in lowered for bad in FORBIDDEN_ASSERTIONS):
        return False
    return any(term in lowered for term in HEDGE_TERMS)


def choose_kind(
    probability: float,
    recent_entry_count: float,
    possession_duration_s: float,
    high_band: float,
) -> InsightKind:
    """Pick the insight kind for a scored window.

    Args:
        probability: Model probability.
        recent_entry_count: Penalty-area entries by this team in the recent past.
        possession_duration_s: How long this team has held the ball.
        high_band: Probability at or above which the stronger wording is used.

    Returns:
        The chosen kind.
    """
    if recent_entry_count >= 2 and possession_duration_s >= 8.0:
        return InsightKind.SUSTAINED_PRESSURE
    if probability >= high_band:
        return InsightKind.ELEVATED_ENTRY_CHANCE
    return InsightKind.BUILDING_THREAT


def headline_for(kind: InsightKind) -> str:
    """Fixed headline for an insight kind."""
    return _HEADLINES[kind]


def extract_facts(
    window: np.ndarray,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    max_facts: int = 3,
) -> tuple[ContextFact, ...]:
    """Derive plainly stated factual context from the scored window.

    These are measurements from the same window the model saw, not predictions,
    so they are phrased without hedging. They give a viewer something concrete
    alongside an inherently uncertain headline.

    Args:
        window: The model input, ``(sequence_length, n_features)``.
        spec: Feature schema used to resolve columns by name.
        max_facts: Maximum number of facts to return.

    Returns:
        Up to ``max_facts`` facts, ordered by how noteworthy they are.
    """
    last = np.asarray(window)[-1]

    def value(name: str) -> float:
        return float(last[spec.index(name)])

    candidates: list[tuple[float, ContextFact]] = []

    ahead = value("attackers_ahead_of_ball")
    if ahead >= 2:
        candidates.append(
            (
                ahead,
                ContextFact(
                    key="attackers_ahead",
                    text=f"{int(ahead)} attackers ahead of the ball",
                    value=ahead,
                ),
            )
        )

    nearest = value("nearest_defender_dist")
    if nearest >= 5.0:
        candidates.append(
            (
                nearest / 5.0,
                ContextFact(
                    key="nearest_defender",
                    text=f"nearest defender {nearest:.0f} m away",
                    value=nearest,
                ),
            )
        )

    entries = value("recent_box_entry_count")
    if entries >= 1:
        plural = "entry" if entries == 1 else "entries"
        candidates.append(
            (
                2.0 + entries,
                ContextFact(
                    key="recent_entries",
                    text=f"{int(entries)} recent penalty-area {plural}",
                    value=entries,
                ),
            )
        )

    line = value("defensive_line_x")
    if line > 5.0:
        candidates.append(
            (
                line / 10.0,
                ContextFact(
                    key="high_line",
                    text=f"defensive line {line:.0f} m beyond halfway",
                    value=line,
                ),
            )
        )

    space = value("space_ahead_of_ball")
    if space >= 12.0:
        candidates.append(
            (
                space / 12.0,
                ContextFact(
                    key="space_ahead",
                    text=f"{space:.0f} m of space ahead of the ball",
                    value=space,
                ),
            )
        )

    possession = value("possession_duration_s")
    if possession >= 12.0:
        candidates.append(
            (
                possession / 12.0,
                ContextFact(
                    key="possession",
                    text=f"{possession:.0f} s of unbroken possession",
                    value=possession,
                ),
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return tuple(fact for _, fact in candidates[:max_facts])


def build_detail(facts: tuple[ContextFact, ...]) -> str:
    """Join factual context into one readable clause.

    Args:
        facts: Facts to include.

    Returns:
        A sentence fragment, or an empty string when there is nothing to add.
    """
    if not facts:
        return ""
    texts = [f.text for f in facts]
    if len(texts) == 1:
        return texts[0].capitalize()
    return (", ".join(texts[:-1]) + f" and {texts[-1]}").capitalize()
