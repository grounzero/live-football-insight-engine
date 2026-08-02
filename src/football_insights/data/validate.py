"""Source data validation.

The engine fails loudly on invalid source data. Silently repairing a feed —
interpolating a gap, clipping a wild coordinate, reordering frames — produces a
model trained on observations that were never made, and the damage is invisible
until it reaches production.

Checks are therefore split in two:

* **Fatal** problems (wrong shapes, unordered frames, coordinates far outside
  the pitch) raise :class:`~football_insights.errors.DataValidationError`.
* **Reportable** problems (a few missing ball samples, brief player dropouts)
  are counted and returned, because real optical tracking always has some and
  refusing to load would make the system useless. They are surfaced in the
  preprocessing report and feed the window validator at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from football_insights.errors import DataValidationError
from football_insights.pitch import DEFAULT_PITCH, Pitch
from football_insights.types import BoolArray, JsonDict

if TYPE_CHECKING:
    from football_insights.domain import Event, MatchTracking

#: Players are routinely tracked a little beyond the touchline; beyond this the
#: coordinate system itself is suspect.
OFF_PITCH_TOLERANCE_M = 5.0

#: Fraction of **in-play** frames allowed to be missing the ball before loading
#: fails.
#:
#: Measuring this over all frames would be misleading. In the Metrica sample
#: data the ball is simply not tracked while play is stopped: 39% of frames in
#: sample game 1 have no ball position, but 68% of those fall in dead-ball
#: periods. So the check is applied to in-play frames when the caller can supply
#: a dead-ball mask, and is reported but not enforced otherwise.
#:
#: The limit is set from what the dataset actually contains rather than from
#: preference. Measured in-play ball-missing rates are 18.0%, 24.6% and 27.2%
#: for sample games 1, 2 and 3. Part of that is genuine tracking dropout and
#: part is the dead-ball mask being approximate — it ends a stoppage at the next
#: on-ball event, while the ball often is not tracked until slightly after that.
#: This gate exists to catch a catastrophically broken feed, not to enforce a
#: quality bar the source never meets; per-window validity, which decides
#: whether anything is actually predicted, is enforced separately by
#: :class:`~football_insights.features.window.RollingWindow`.
MAX_BALL_MISSING_RATIO_IN_PLAY = 0.40

#: Fraction of player samples allowed to sit beyond OFF_PITCH_TOLERANCE_M before
#: loading fails. A handful is normal; two per cent means the coordinate system
#: or the pitch dimensions are wrong, not that players ran off the field.
MAX_OFF_PITCH_PLAYER_RATIO = 0.02

#: The same idea for the ball, which legitimately leaves the pitch and so gets a
#: looser bound.
MAX_OFF_PITCH_BALL_RATIO = 0.05


@dataclass(slots=True)
class ValidationReport:
    """What validation found, for the preprocessing artifact."""

    match_id: str
    n_frames: int
    frame_rate: float
    duration_s: float
    ball_missing: int = 0
    ball_missing_ratio: float = 0.0
    ball_missing_ratio_in_play: float | None = None
    player_missing_ratio: float = 0.0
    duplicate_frames: int = 0
    frame_gaps: int = 0
    largest_gap_frames: int = 0
    off_pitch_samples: int = 0
    events_total: int = 0
    events_dropped: int = 0
    warnings: list[str] = field(default_factory=list[str])

    def to_dict(self) -> JsonDict:
        """Serialisable form."""
        return {
            "match_id": self.match_id,
            "n_frames": self.n_frames,
            "frame_rate": self.frame_rate,
            "duration_s": round(self.duration_s, 2),
            "ball_missing": self.ball_missing,
            "ball_missing_ratio": round(self.ball_missing_ratio, 5),
            "ball_missing_ratio_in_play": (
                None
                if self.ball_missing_ratio_in_play is None
                else round(self.ball_missing_ratio_in_play, 5)
            ),
            "player_missing_ratio": round(self.player_missing_ratio, 5),
            "duplicate_frames": self.duplicate_frames,
            "frame_gaps": self.frame_gaps,
            "largest_gap_frames": self.largest_gap_frames,
            "off_pitch_samples": self.off_pitch_samples,
            "events_total": self.events_total,
            "events_dropped": self.events_dropped,
            "warnings": self.warnings,
        }


def validate_tracking(
    tracking: MatchTracking,
    match_id: str,
    pitch: Pitch = DEFAULT_PITCH,
    in_play: np.ndarray | None = None,
) -> ValidationReport:
    """Validate a parsed match and report its data quality.

    Args:
        tracking: The parsed tracking data.
        match_id: Identifier used in error messages and the report.
        pitch: Pitch dimensions used for plausibility checks.
        in_play: Boolean mask of frames with the ball in open play. When given,
            the ball-coverage limit is applied to these frames only, which is
            the number that actually matters; missing ball during a stoppage is
            expected in optical tracking and is reported rather than enforced.

    Returns:
        A report of non-fatal findings.

    Raises:
        DataValidationError: On a problem that makes the data unusable.
    """
    report = ValidationReport(
        match_id=match_id,
        n_frames=tracking.n_frames,
        frame_rate=tracking.frame_rate,
        duration_s=tracking.duration_s,
    )

    check = _Check(tracking=tracking, match_id=match_id, pitch=pitch)

    # Rule order is the contract: each rule may assume the previous ones held,
    # and the warnings accumulate in this order.
    _require_usable_shape(check)
    _check_frame_ordering(check, report)
    _check_timestamps(check)

    # np.all, not ndarray.all: numpy's stubs before 2.5 type the method's
    # axis-given overload as `np.bool | NDArray[np.bool]`, which fails a strict
    # check on the 3.11 leg (2.5 needs 3.12, so 3.11 resolves numpy 2.4).
    ball_finite: BoolArray = np.all(np.isfinite(tracking.ball_xy), axis=1)
    _check_ball_coverage(check, report, ball_finite, in_play)
    _check_player_positions(check, report)
    _check_ball_positions(check, ball_finite)
    return report


@dataclass(frozen=True, slots=True)
class _Check:
    """Inputs shared by every validation rule."""

    tracking: MatchTracking
    match_id: str
    pitch: Pitch


def _require_usable_shape(check: _Check) -> None:
    """Reject input that cannot be validated at all.

    Raises:
        DataValidationError: If the match has no frames, no positive frame rate,
            or a team with no tracked players.
    """
    tracking, match_id = check.tracking, check.match_id
    if tracking.n_frames == 0:
        msg = f"{match_id}: tracking contains no frames"
        raise DataValidationError(msg)
    if tracking.frame_rate <= 0:
        msg = f"{match_id}: frame rate must be positive, got {tracking.frame_rate}"
        raise DataValidationError(msg)
    if not tracking.home_players or not tracking.away_players:
        msg = f"{match_id}: both teams must have tracked players"
        raise DataValidationError(msg)


def _check_frame_ordering(check: _Check, report: ValidationReport) -> None:
    """Frames must advance within each period; count duplicates and gaps.

    Frames going backwards is fatal — the file is not what it claims to be —
    while duplicates and gaps are normal in optical tracking and are reported.

    Raises:
        DataValidationError: If periods or frames go backwards.
    """
    tracking, match_id = check.tracking, check.match_id
    periods, frames = tracking.period, tracking.frame
    if np.any(np.diff(periods) < 0):
        msg = f"{match_id}: periods are not monotonically increasing"
        raise DataValidationError(msg)

    for period in np.unique(periods):
        mask: BoolArray = periods == period
        duplicates, gap_count, largest_gap = _period_frame_stats(
            frames[mask], int(period), match_id
        )
        report.duplicate_frames += duplicates
        report.frame_gaps += gap_count
        report.largest_gap_frames = max(report.largest_gap_frames, largest_gap)

    if report.duplicate_frames:
        report.warnings.append(f"{report.duplicate_frames} duplicate frame indices")
    if report.frame_gaps:
        report.warnings.append(
            f"{report.frame_gaps} gaps in the frame sequence, largest "
            f"{report.largest_gap_frames} frames"
        )


def _period_frame_stats(seq: np.ndarray, period: int, match_id: str) -> tuple[int, int, int]:
    """Summarise one period's frame sequence.

    Returns:
        ``(duplicate count, gap count, largest gap in frames)``. A gap is
        reported as the number of *missing* frames, so consecutive frames give
        zero.

    Raises:
        DataValidationError: If the sequence goes backwards, which means the
            file is not the ordered stream it claims to be.
    """
    deltas = np.diff(seq)
    if np.any(deltas < 0):
        first = int(np.flatnonzero(deltas < 0)[0])
        msg = (
            f"{match_id}: frames go backwards in period {period} at index {first} "
            f"({seq[first]} -> {seq[first + 1]})"
        )
        raise DataValidationError(msg)
    gaps = deltas[deltas > 1]
    largest = int(gaps.max() - 1) if gaps.size else 0
    return int(np.count_nonzero(deltas == 0)), int(gaps.size), largest


def _check_timestamps(check: _Check) -> None:
    """Timestamps must not go backwards.

    Raises:
        DataValidationError: If they do.
    """
    if np.any(np.diff(check.tracking.time_s) < 0):
        msg = f"{check.match_id}: timestamps are not monotonically increasing"
        raise DataValidationError(msg)


def _check_ball_coverage(
    check: _Check,
    report: ValidationReport,
    ball_finite: BoolArray,
    in_play: np.ndarray | None,
) -> None:
    """Measure ball coverage, enforcing the limit only over in-play frames.

    Missing ball during a stoppage is expected and is reported; missing ball
    while the ball is live past :data:`MAX_BALL_MISSING_RATIO_IN_PLAY` means the
    feed is unusable. See that constant for why the threshold sits where it does.

    Raises:
        DataValidationError: If in-play coverage is below the limit.
    """
    report.ball_missing = int((~ball_finite).sum())
    report.ball_missing_ratio = float(report.ball_missing / check.tracking.n_frames)
    if report.ball_missing:
        report.warnings.append(
            f"ball missing in {report.ball_missing} frames "
            f"({report.ball_missing_ratio:.2%} of all frames)"
        )

    if in_play is None or not in_play.any():
        return
    live_missing = int((~ball_finite & in_play).sum())
    report.ball_missing_ratio_in_play = float(live_missing / int(in_play.sum()))
    if report.ball_missing_ratio_in_play > MAX_BALL_MISSING_RATIO_IN_PLAY:
        msg = (
            f"{check.match_id}: ball position missing in "
            f"{report.ball_missing_ratio_in_play:.1%} of in-play frames, above "
            f"the {MAX_BALL_MISSING_RATIO_IN_PLAY:.0%} limit; the source data "
            "is unusable"
        )
        raise DataValidationError(msg)


def _check_player_positions(check: _Check, report: ValidationReport) -> None:
    """Measure player coverage and reject implausible coordinates.

    An implausible coordinate is not a dropout: it means the units or the
    coordinate convention are wrong, which would corrupt every feature.

    Raises:
        DataValidationError: If too many players lie outside the pitch.
    """
    total_slots = 0
    missing_slots = 0
    off_pitch = 0
    for xy in (check.tracking.home_xy, check.tracking.away_xy):
        finite: BoolArray = np.all(np.isfinite(xy), axis=2)
        total_slots += finite.size
        missing_slots += int((~finite).sum())
        on = check.pitch.is_on_pitch(xy, tolerance=OFF_PITCH_TOLERANCE_M)
        off_pitch += int((finite & ~on).sum())

    report.player_missing_ratio = float(missing_slots / total_slots) if total_slots else 0.0
    report.off_pitch_samples = off_pitch

    off_pitch_ratio = off_pitch / total_slots if total_slots else 0.0
    if off_pitch_ratio > MAX_OFF_PITCH_PLAYER_RATIO:
        msg = (
            f"{check.match_id}: {off_pitch_ratio:.1%} of player positions lie more than "
            f"{OFF_PITCH_TOLERANCE_M} m outside the pitch. The coordinate system or "
            "pitch dimensions are probably wrong."
        )
        raise DataValidationError(msg)
    if off_pitch:
        report.warnings.append(f"{off_pitch} player samples beyond the touchline")


def _check_ball_positions(check: _Check, ball_finite: BoolArray) -> None:
    """Reject a ball that is repeatedly far outside the pitch.

    Raises:
        DataValidationError: If too many tracked ball samples are implausible.
    """
    ball_on = check.pitch.is_on_pitch(check.tracking.ball_xy, tolerance=OFF_PITCH_TOLERANCE_M)
    ball_off = int((ball_finite & ~ball_on).sum())
    if ball_off / check.tracking.n_frames > MAX_OFF_PITCH_BALL_RATIO:
        msg = (
            f"{check.match_id}: {ball_off} ball samples lie far outside the pitch; "
            "check the coordinate convention"
        )
        raise DataValidationError(msg)


def _events_within_tracking(
    events: tuple[Event, ...], tracking: MatchTracking
) -> tuple[tuple[Event, ...], int, int]:
    """Select events whose start frame the tracking actually covers.

    Returns:
        ``(kept events, first frame, last frame)``. The bounds are returned so
        the caller can name them in its messages.
    """
    lo = int(tracking.frame.min())
    hi = int(tracking.frame.max())
    return tuple(e for e in events if lo <= e.start_frame <= hi), lo, hi


def validate_events(
    events: tuple[Event, ...],
    tracking: MatchTracking,
    match_id: str,
    report: ValidationReport,
) -> tuple[Event, ...]:
    """Check events against the tracking they are meant to align with.

    Events referring to frames outside the tracking range cannot be aligned and
    are dropped with a count, rather than silently clamped to the nearest frame,
    which would fabricate a timestamp.

    Args:
        events: Parsed events.
        tracking: The match's tracking data.
        match_id: Identifier for error messages.
        report: Report to record findings in.

    Returns:
        The events that align with the tracking.

    Raises:
        DataValidationError: If no events align at all.
    """
    report.events_total = len(events)
    if not events:
        msg = f"{match_id}: no events parsed; possession cannot be derived"
        raise DataValidationError(msg)

    kept, lo, hi = _events_within_tracking(events, tracking)
    report.events_dropped = len(events) - len(kept)
    if report.events_dropped:
        report.warnings.append(
            f"{report.events_dropped} events fall outside the tracking frame range "
            f"[{lo}, {hi}] and were dropped"
        )
    if not kept:
        msg = (
            f"{match_id}: no events fall inside the tracking frame range [{lo}, {hi}]; "
            "the event and tracking files are probably from different matches"
        )
        raise DataValidationError(msg)

    out_of_order = sum(1 for a, b in pairwise(kept) if b.start_frame < a.start_frame)
    if out_of_order:
        report.warnings.append(f"{out_of_order} events arrived out of order and were sorted")
    return tuple(sorted(kept, key=lambda e: (e.period, e.start_frame)))
