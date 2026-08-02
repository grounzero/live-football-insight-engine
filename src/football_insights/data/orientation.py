"""Attacking-direction inference.

Getting playing direction wrong inverts every spatial feature in the system,
and does so silently: the pipeline runs, the model trains, and the results are
meaningless. So direction is never taken from a single signal. Several
independent signals vote, the decision and its evidence are written to an audit
artifact, and material disagreement stops preprocessing rather than being
resolved by a rule of thumb.

Evidence hierarchy, strongest first:

============  =========================================  =========================
Tier          Signal                                     Availability
============  =========================================  =========================
1             ``attack_direction_first_half`` metadata   EPTS matches only
2             Mean pass and carry progression            All matches, high volume
3             Goalkeeper position; team centroid depth   All matches, all frames
4             Shot geometry                              All matches, low volume
============  =========================================  =========================

The tiers exist because availability and reliability differ sharply. Sample game
3 declares direction outright *and* labels goalkeepers; sample games 1 and 2
declare neither. That asymmetry is useful rather than awkward: game 3 is a
ground-truth fixture for the inference used on the other two, and
``tests/unit/test_orientation.py`` asserts tiers 2 to 4 reproduce its declared
truth.

Two structural facts are enforced regardless of the vote:

* the two teams must attack opposite ends in the same period;
* a team's direction must flip between the first and second half.

Either violation is a hard failure. They are properties of football, not
statistical tendencies, so a fit that breaks them is wrong however confident it
looks.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import numpy as np

from football_insights.domain import (
    AttackDirection,
    Event,
    EventType,
    MatchTracking,
    Orientation,
    PlayerRef,
    Team,
)
from football_insights.errors import OrientationError
from football_insights.types import BoolArray, JsonDict

if TYPE_CHECKING:
    from pathlib import Path

#: Weight per evidence tier. Tier 1 dominates because it is a statement from the
#: provider rather than an inference, but it does not silence the others: a
#: high-volume signal contradicting the metadata means something is wrong with
#: the pairing of files and is worth stopping for.
TIER_WEIGHTS: Final[dict[int, float]] = {1: 1.0, 2: 0.75, 3: 0.65, 4: 0.35}

#: Below this weighted agreement the evidence is treated as contradictory.
MIN_AGREEMENT: Final = 0.70

#: A signal with at least this much weighted strength counts as "high volume"
#: when checking whether it contradicts declared metadata.
HIGH_VOLUME_STRENGTH: Final = 0.30


@dataclass(frozen=True, slots=True)
class DirectionSignal:
    """One piece of evidence about a team's attacking direction."""

    name: str
    tier: int
    direction: AttackDirection
    #: Confidence in ``[0, 1]``: how far from ambiguous this measurement is.
    margin: float
    sample_size: int
    detail: str

    @property
    def strength(self) -> float:
        """Weighted contribution to the vote."""
        return TIER_WEIGHTS[self.tier] * self.margin

    def to_dict(self) -> JsonDict:
        """Serialisable form for the audit artifact."""
        payload = asdict(self)
        payload["direction"] = self.direction.value
        payload["strength"] = round(self.strength, 4)
        payload["margin"] = round(self.margin, 4)
        return payload


@dataclass(frozen=True, slots=True)
class DirectionDecision:
    """The chosen direction for one team in one period, with its evidence."""

    period: int
    team: Team
    direction: AttackDirection
    agreement: float
    source: str
    signals: tuple[DirectionSignal, ...]
    override_reason: str | None = None

    def to_dict(self) -> JsonDict:
        """Serialisable form for the audit artifact."""
        return {
            "period": self.period,
            "team": self.team.value,
            "direction": self.direction.value,
            "agreement": round(self.agreement, 4),
            "source": self.source,
            "override_reason": self.override_reason,
            "signals": [s.to_dict() for s in self.signals],
        }


def _margin(value: float, scale: float) -> float:
    """Map a signed measurement to a confidence in ``[0, 1]``.

    Args:
        value: The measurement; its sign gives the direction.
        scale: Magnitude at which confidence saturates.

    Returns:
        ``min(1, |value| / scale)``.
    """
    return float(min(1.0, abs(value) / scale)) if scale > 0 else 0.0


def _pass_progression(events: tuple[Event, ...], period: int, team: Team) -> DirectionSignal | None:
    """Tier 2: teams move the ball toward the goal they are attacking.

    Individually a pass says little, but averaged over hundreds the drift is
    unambiguous, which makes this the most dependable inferred signal.
    """
    deltas = [
        e.end_xy[0] - e.start_xy[0]
        for e in events
        if e.period == period
        and e.team is team
        and e.type in (EventType.PASS, EventType.CARRY)
        and e.start_xy is not None
        and e.end_xy is not None
    ]
    if len(deltas) < 20:
        return None
    mean = float(np.mean(deltas))
    return DirectionSignal(
        name="pass_progression",
        tier=2,
        direction=AttackDirection.from_sign(mean),
        margin=_margin(mean, 1.5),
        sample_size=len(deltas),
        detail=f"mean pass/carry progression {mean:+.2f} m over {len(deltas)} events",
    )


def _goalkeeper_position(
    tracking: MatchTracking, period: int, team: Team, gk_index: int | None
) -> DirectionSignal | None:
    """Tier 3: a team attacks away from its own goalkeeper.

    Uses every frame of the period rather than a handful of events, so it is
    robust even in a half with almost no attacking output.
    """
    if gk_index is None:
        return None
    mask: BoolArray = tracking.period == period
    if not mask.any():
        return None
    xs = tracking.team_xy(team)[mask, gk_index, 0]
    xs = xs[np.isfinite(xs)]
    if xs.size < 100:
        return None
    mean = float(np.mean(xs))
    # The keeper sits near their own goal, so the team attacks the other way.
    return DirectionSignal(
        name="goalkeeper_position",
        tier=3,
        direction=AttackDirection.from_sign(-mean),
        margin=_margin(mean, 25.0),
        sample_size=int(xs.size),
        detail=f"goalkeeper mean x {mean:+.1f} m over {xs.size} frames",
    )


def _team_centroid(tracking: MatchTracking, period: int, team: Team) -> DirectionSignal | None:
    """Tier 3: a team's shape is anchored near the goal it defends.

    Over a whole half a team is spread from its own goal forward, with the
    keeper and the defensive line holding the rear. The side defending the ``-x``
    goal therefore has the lower mean x, and attacks ``+x`` — the team attacks
    *away* from its own centroid, not toward it.

    Weaker than the goalkeeper signal and weighted as such, but it needs no
    player identification at all, so it still works when the keeper cannot be
    established.
    """
    mask = tracking.period == period
    if not mask.any():
        return None
    own = tracking.team_xy(team)[mask][..., 0]
    other = tracking.team_xy(team.opponent)[mask][..., 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        own_mean = float(np.nanmean(own))
        other_mean = float(np.nanmean(other))
    if not (np.isfinite(own_mean) and np.isfinite(other_mean)):
        return None
    diff = own_mean - other_mean
    return DirectionSignal(
        name="team_centroid",
        tier=3,
        direction=AttackDirection.from_sign(-diff),
        margin=_margin(diff, 6.0),
        sample_size=int(mask.sum()),
        detail=(
            f"centroid {own_mean:+.1f} m vs opponent {other_mean:+.1f} m (deeper side attacks +x)"
        ),
    )


def _shot_geometry(events: tuple[Event, ...], period: int, team: Team) -> DirectionSignal | None:
    """Tier 4: shots are taken toward the goal being attacked.

    The most direct evidence available and the weakest in practice, because a
    team may take only two or three shots in a half.
    """
    xs = [
        e.start_xy[0]
        for e in events
        if e.period == period and e.team is team and e.type is EventType.SHOT and e.start_xy
    ]
    if not xs:
        return None
    mean = float(np.mean(xs))
    return DirectionSignal(
        name="shot_geometry",
        tier=4,
        direction=AttackDirection.from_sign(mean),
        margin=_margin(mean, 25.0) * min(1.0, len(xs) / 5.0),
        sample_size=len(xs),
        detail=f"mean shot x {mean:+.1f} m over {len(xs)} shots",
    )


def identify_goalkeepers(
    tracking: MatchTracking, players: tuple[PlayerRef, ...], team: Team
) -> tuple[tuple[PlayerRef, ...], int | None]:
    """Return players with the goalkeeper marked, and that keeper's column.

    When the source declares a keeper it is trusted. Otherwise the keeper is
    inferred as the outfield-extreme player: the one whose mean position is
    furthest from the centre along x, averaged over the match. Which of the two
    is used is recorded on the player, so the audit artifact never implies the
    source said something it did not.
    """
    declared = [i for i, p in enumerate(players) if p.is_goalkeeper]
    if declared:
        return players, declared[0]

    xy = tracking.team_xy(team)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(xy[..., 0], axis=0)
    if not np.isfinite(means).any():
        return players, None
    # A keeper spends the match near one goal, so the *magnitude* of their mean
    # x is the largest of the team once both halves are averaged separately.
    scores = np.full(means.shape, -np.inf)
    for period in np.unique(tracking.period):
        mask = tracking.period == period
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            period_means = np.nanmean(xy[mask][..., 0], axis=0)
        finite = np.isfinite(period_means)
        scores[finite] = np.maximum(scores[finite], np.abs(period_means[finite]))
    index = int(np.argmax(scores))
    updated = tuple(
        PlayerRef(
            player_id=p.player_id,
            team=p.team,
            shirt_number=p.shirt_number,
            position_type=p.position_type,
            is_goalkeeper=(i == index),
            goalkeeper_source="inferred" if i == index else p.goalkeeper_source,
        )
        for i, p in enumerate(players)
    )
    return updated, index


@dataclass(frozen=True, slots=True)
class _Slot:
    """One ``(period, team)`` pair being decided."""

    match_id: str
    period: int
    team: Team

    @property
    def override_key(self) -> str:
        """Key used to look this slot up in ``direction_overrides``."""
        return f"{self.match_id}:{self.period}:{self.team.value}"


@dataclass(frozen=True, slots=True)
class _Evidence:
    """Everything the direction vote reads, fixed for the whole match.

    Bundled so the per-slot helpers take a context and a slot rather than
    threading six loop-invariant values through each call.
    """

    tracking: MatchTracking
    events: tuple[Event, ...]
    declared: dict[tuple[int, Team], AttackDirection]
    gk_index: dict[Team, int | None]


def _declared_signal(evidence: _Evidence, slot: _Slot) -> DirectionSignal | None:
    """Tier 1: the provider states the direction outright."""
    stated = evidence.declared.get((slot.period, slot.team))
    if stated is None:
        return None
    return DirectionSignal(
        name="provider_metadata",
        tier=1,
        direction=stated,
        margin=1.0,
        sample_size=1,
        detail=f"source declares {stated.value}",
    )


def _collect_signals(evidence: _Evidence, slot: _Slot) -> tuple[DirectionSignal, ...]:
    """Every available piece of evidence for one slot, strongest tier first.

    Collection is kept separate from resolution so that adding or removing a
    signal cannot accidentally change how the vote is settled.
    """
    period, team = slot.period, slot.team
    candidates = (
        _declared_signal(evidence, slot),
        _pass_progression(evidence.events, period, team),
        _goalkeeper_position(evidence.tracking, period, team, evidence.gk_index[team]),
        _team_centroid(evidence.tracking, period, team),
        _shot_geometry(evidence.events, period, team),
    )
    return tuple(signal for signal in candidates if signal is not None)


def _tally(signals: tuple[DirectionSignal, ...]) -> tuple[AttackDirection, float]:
    """Settle the weighted vote.

    Returns:
        The winning direction and the share of total strength behind it. A tie
        resolves to ``+x``, which only arises when the evidence is already too
        weak to pass :data:`MIN_AGREEMENT`.
    """
    totals: dict[AttackDirection, float] = {
        AttackDirection.POSITIVE_X: 0.0,
        AttackDirection.NEGATIVE_X: 0.0,
    }
    for signal in signals:
        totals[signal.direction] += signal.strength
    chosen = max(totals, key=lambda d: totals[d])
    total_strength = sum(totals.values())
    agreement = totals[chosen] / total_strength if total_strength > 0 else 0.0
    return chosen, agreement


def _is_contradictory(
    signals: tuple[DirectionSignal, ...], chosen: AttackDirection, agreement: float
) -> bool:
    """Whether the evidence is too weak or too divided to act on.

    Two distinct failures, both fatal:

    * the vote itself is close, so no direction is well supported;
    * a high-volume signal contradicts the provider's *declared* direction,
      which usually means the tracking and metadata files describe different
      matches. That is worth stopping for even when agreement looks high,
      because tier 1 carries enough weight to win the vote on its own.
    """
    if agreement < MIN_AGREEMENT:
        return True
    if not any(s.tier == 1 for s in signals):
        return False
    return any(s.direction is not chosen and s.strength >= HIGH_VOLUME_STRENGTH for s in signals)


def _decide_slot(
    evidence: _Evidence, slot: _Slot, override: AttackDirection | None, reason: str | None
) -> DirectionDecision:
    """Decide one slot, or stop preprocessing if the evidence will not support it.

    Raises:
        OrientationError: If no evidence exists, or if the signals contradict
            each other or the provider's own metadata.
    """
    signals = _collect_signals(evidence, slot)

    if override is not None:
        return DirectionDecision(
            period=slot.period,
            team=slot.team,
            direction=override,
            agreement=1.0,
            source="override",
            signals=signals,
            override_reason=reason,
        )

    if not signals:
        msg = (
            f"{slot.match_id}: no evidence available for period {slot.period}, "
            f"{slot.team.value}. Set a direction_overrides entry for "
            f"{slot.override_key!r} with a reason if this match genuinely cannot "
            "be inferred."
        )
        raise OrientationError(msg)

    chosen, agreement = _tally(signals)
    if _is_contradictory(signals, chosen, agreement):
        _raise_disagreement(slot, chosen, agreement, signals)

    source = "metadata" if any(s.tier == 1 for s in signals) else "inferred"
    return DirectionDecision(
        period=slot.period,
        team=slot.team,
        direction=chosen,
        agreement=agreement,
        source=source,
        signals=signals,
    )


def _direction_report(
    match_id: str,
    periods: list[int],
    decisions: list[DirectionDecision],
    goalkeepers: tuple[tuple[PlayerRef, ...], tuple[PlayerRef, ...]],
) -> JsonDict:
    """Assemble the audit artifact written to ``direction_report.json``."""
    home_players, away_players = goalkeepers
    return {
        "match_id": match_id,
        "periods": periods,
        "min_agreement": MIN_AGREEMENT,
        "tier_weights": {str(k): v for k, v in TIER_WEIGHTS.items()},
        "goalkeepers": {
            "home": _gk_description(home_players),
            "away": _gk_description(away_players),
        },
        "decisions": [d.to_dict() for d in decisions],
    }


def infer_orientation(
    tracking: MatchTracking,
    events: tuple[Event, ...],
    match_id: str,
    declared: dict[tuple[int, Team], AttackDirection] | None = None,
    overrides: dict[str, str] | None = None,
    override_reasons: dict[str, str] | None = None,
) -> tuple[Orientation, tuple[PlayerRef, ...], tuple[PlayerRef, ...]]:
    """Establish attacking direction for every period and team.

    Args:
        tracking: Parsed tracking in canonical coordinates.
        events: Parsed events.
        match_id: Identifier used in overrides, errors and the report.
        declared: Direction stated by the source, keyed by period and team.
        overrides: Manual overrides keyed ``"<match_id>:<period>:<team>"`` with
            a value of ``"+x"`` or ``"-x"``.
        override_reasons: Justification per override key, recorded in the report.

    Returns:
        The orientation with its audit report, plus both teams' player
        references with goalkeepers marked.

    Raises:
        OrientationError: If evidence is contradictory, if the two teams are
            found to attack the same end, or if a team fails to change ends
            between halves.
    """
    overrides = overrides or {}
    override_reasons = override_reasons or {}
    home_players, home_gk = identify_goalkeepers(tracking, tracking.home_players, Team.HOME)
    away_players, away_gk = identify_goalkeepers(tracking, tracking.away_players, Team.AWAY)

    evidence = _Evidence(
        tracking=tracking,
        events=events,
        declared=declared or {},
        gk_index={Team.HOME: home_gk, Team.AWAY: away_gk},
    )
    periods = [int(p) for p in np.unique(tracking.period)]

    decisions = [
        _decide_slot(
            evidence,
            slot,
            override=(
                AttackDirection(overrides[slot.override_key])
                if slot.override_key in overrides
                else None
            ),
            reason=override_reasons.get(slot.override_key),
        )
        for period in periods
        for slot in (_Slot(match_id, period, Team.HOME), _Slot(match_id, period, Team.AWAY))
    ]

    directions = {(d.period, d.team): d.direction for d in decisions}
    _check_structure(match_id, directions, periods, decisions)

    report = _direction_report(match_id, periods, decisions, (home_players, away_players))
    return Orientation(directions=directions, report=report), home_players, away_players


def _gk_description(players: tuple[PlayerRef, ...]) -> JsonDict:
    """Describe which player is treated as the keeper and how that was decided."""
    for player in players:
        if player.is_goalkeeper:
            return {
                "player_id": player.player_id,
                "shirt_number": player.shirt_number,
                "source": player.goalkeeper_source,
            }
    return {"player_id": None, "source": "none"}


def _raise_disagreement(
    slot: _Slot,
    chosen: AttackDirection,
    agreement: float,
    signals: tuple[DirectionSignal, ...],
) -> None:
    """Stop preprocessing with every signal listed."""
    lines = [
        f"{slot.match_id}: contradictory evidence for attacking direction in period "
        f"{slot.period} for the {slot.team.value} team.",
        f"  best guess {chosen.value} with weighted agreement {agreement:.2f} "
        f"(minimum {MIN_AGREEMENT:.2f})",
    ]
    lines.extend(
        f"  tier {s.tier} {s.name}: {s.direction.value} (strength {s.strength:.2f}) — {s.detail}"
        for s in sorted(signals, key=lambda s: s.tier)
    )
    lines.append(
        f"  To proceed anyway, set direction_overrides[{slot.override_key!r}] with a "
        "direction_override_reasons entry explaining why."
    )
    raise OrientationError("\n".join(lines))


def _check_opposite_ends(
    match_id: str,
    directions: dict[tuple[int, Team], AttackDirection],
    periods: list[int],
    decisions: list[DirectionDecision],
) -> None:
    """Both teams cannot attack the same goal in the same period."""
    for period in periods:
        home = directions[(period, Team.HOME)]
        if home is not directions[(period, Team.AWAY)]:
            continue
        overridden = any(d.period == period and d.source == "override" for d in decisions)
        hint = (
            " An override forced this; check its reason."
            if overridden
            else " Check the source files are from the same match."
        )
        msg = (
            f"{match_id}: both teams were found to attack {home.value} in period "
            f"{period}. Two teams cannot attack the same goal.{hint}"
        )
        raise OrientationError(msg)


def _check_ends_swap_at_half_time(
    match_id: str,
    directions: dict[tuple[int, Team], AttackDirection],
    periods: list[int],
) -> None:
    """A team must attack the opposite way in the second half from the first.

    Only regulation periods 1 and 2 are compared; extra time restarts the
    pattern and is not constrained here.
    """
    for team in (Team.HOME, Team.AWAY):
        for first, second in pairwise(periods):
            if {first, second} != {1, 2}:
                continue
            if directions[(first, team)] is not directions[(second, team)]:
                continue
            msg = (
                f"{match_id}: the {team.value} team was found to attack "
                f"{directions[(first, team)].value} in both period {first} and "
                f"period {second}. Teams change ends at half time, so at least "
                "one period has been inferred wrongly."
            )
            raise OrientationError(msg)


def _check_structure(
    match_id: str,
    directions: dict[tuple[int, Team], AttackDirection],
    periods: list[int],
    decisions: list[DirectionDecision],
) -> None:
    """Enforce the two facts that hold in every football match.

    These are checked after the vote because they are not evidence to be
    weighed: a result that breaks them is wrong no matter how the signals fell.
    """
    _check_opposite_ends(match_id, directions, periods, decisions)
    _check_ends_swap_at_half_time(match_id, directions, periods)


def write_direction_report(orientation: Orientation, path: Path) -> None:
    """Write the audit artifact.

    Written before any downstream failure so a rejected match can still be
    diagnosed from the report rather than only from a stack trace.

    Args:
        orientation: The orientation to record.
        path: Destination JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(orientation.report, indent=2, sort_keys=True) + "\n")
