"""Penalty-area entry labelling and episode grouping.

**The target.** At prediction time ``t``, positive if and only if the ball —
while the attacking team is in possession — crosses from outside into the
opposition penalty area at some point in ``(t, t + H]``.

This target was chosen from the data rather than from preference. Across the
three sample matches there are roughly 24 shots per match but 50 to 60
penalty-area entries, and a held-out match containing 24 positives cannot
support a precision estimate anyone should act on. Its limitations are real and
documented in the model card: a box entry is not danger, it excludes long-range
shots, and possession is an approximation derived from event annotation.

**Why episodes exist.** With a 0.5 s stride and a 5 s horizon, a single entry
produces about ten positive windows that overlap almost completely. Treating
those as ten independent observations would inflate every metric and every
confidence interval. Ground-truth entries close together in time are therefore
merged into one *episode*, and the evaluation harness scores episodes as well as
windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from football_insights.domain import Team
from football_insights.features.frame_features import box_entry_mask
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from football_insights.config import EpisodeSettings, WindowSettings
    from football_insights.domain import Orientation


@dataclass(frozen=True, slots=True)
class Episode:
    """A merged ground-truth penalty-area entry."""

    match_id: str
    team: Team
    period: int
    #: Time of the first entry in the merged group.
    entry_time_s: float
    #: Number of raw entries merged into this episode.
    n_entries: int


@dataclass(frozen=True, slots=True)
class LabelledWindows:
    """Prediction samples for one match.

    Attributes:
        end_index: Row index into the match's frame arrays for each sample's
            prediction instant.
        label: 1 when an entry occurs within the horizon, else 0.
        episode_id: Index into ``episodes`` for positive samples, ``-1`` otherwise.
        cluster_id: Possession-sequence identifier used as the bootstrap
            resampling unit. Windows in the same cluster are correlated and must
            never be resampled independently.
    """

    match_id: str
    end_index: np.ndarray
    time_s: np.ndarray
    period: np.ndarray
    attacking_team: np.ndarray
    label: np.ndarray
    episode_id: np.ndarray
    cluster_id: np.ndarray
    episodes: tuple[Episode, ...]

    def __len__(self) -> int:
        """Number of samples."""
        return int(self.label.shape[0])

    @property
    def positive_rate(self) -> float:
        """Fraction of samples labelled positive."""
        return float(self.label.mean()) if len(self) else 0.0

    def summary(self) -> JsonDict:
        """Class-balance summary for the preprocessing report."""
        return {
            "match_id": self.match_id,
            "samples": len(self),
            "positives": int(self.label.sum()),
            "positive_rate": round(self.positive_rate, 5),
            "episodes": len(self.episodes),
            "clusters": len(np.unique(self.cluster_id)),
            "episodes_by_team": {
                team.value: sum(1 for e in self.episodes if e.team is team)
                for team in (Team.HOME, Team.AWAY)
            },
        }


def find_entries(
    ball_xy: np.ndarray,
    times_s: np.ndarray,
    periods: np.ndarray,
    possession_team: np.ndarray,
    orientation: Orientation,
) -> list[tuple[float, Team, int]]:
    """Locate every penalty-area entry in a match.

    An entry is a rising edge of "ball inside the attacking penalty area",
    evaluated against the possessing team's attacking direction. Detecting it
    from tracking rather than from event annotation gives the crossing instant
    at full 25 Hz resolution, which matters because warning time is reported.

    Args:
        ball_xy: Ball positions in canonical coordinates, ``(n_frames, 2)``.
        times_s: Match time per frame.
        periods: Period per frame.
        possession_team: Possessing team per frame, ``0`` home, ``1`` away,
            ``-1`` unknown.
        orientation: Attacking direction per period and team.

    Returns:
        Tuples of ``(entry time, attacking team, frame index)``.
    """
    n = ball_xy.shape[0]
    inside = np.zeros(n, dtype=bool)
    for period in np.unique(periods):
        for code, team in ((0, Team.HOME), (1, Team.AWAY)):
            mask = (periods == period) & (possession_team == code)
            if not mask.any():
                continue
            sign = orientation.direction(int(period), team).sign
            inside[mask] = box_entry_mask(ball_xy[mask], sign)

    entries: list[tuple[float, Team, int]] = []
    previous = False
    for i in range(n):
        current = bool(inside[i])
        if current and not previous:
            code = int(possession_team[i])
            if code in (0, 1):
                entries.append((float(times_s[i]), Team.HOME if code == 0 else Team.AWAY, i))
        previous = current
    return entries


def merge_episodes(
    entries: list[tuple[float, Team, int]],
    match_id: str,
    periods: np.ndarray,
    merge_gap_s: float,
) -> tuple[Episode, ...]:
    """Collapse closely spaced entries by the same team into single episodes.

    A cross cleared and immediately re-delivered is one attacking passage, not
    two independent chances to be right, and scoring it twice would flatter any
    model that fires once and keeps firing.

    Args:
        entries: Raw entries from :func:`find_entries`.
        match_id: Identifier stored on each episode.
        periods: Period per frame, used to stamp the episode.
        merge_gap_s: An entry within this long of the episode's **first** entry
            joins it. Measuring from the first entry rather than the previous
            one keeps episodes bounded: chaining pairwise would let a sequence
            of entries eight seconds apart merge into a single episode spanning
            an entire half.

    Returns:
        Merged episodes in time order.
    """
    episodes: list[Episode] = []
    for time_s, team, index in entries:
        if (
            episodes
            and episodes[-1].team is team
            and time_s - episodes[-1].entry_time_s <= merge_gap_s
        ):
            last = episodes[-1]
            episodes[-1] = Episode(
                match_id=last.match_id,
                team=last.team,
                period=last.period,
                entry_time_s=last.entry_time_s,
                n_entries=last.n_entries + 1,
            )
            continue
        episodes.append(
            Episode(
                match_id=match_id,
                team=team,
                period=int(periods[index]),
                entry_time_s=time_s,
                n_entries=1,
            )
        )
    return tuple(episodes)


def possession_clusters(possession_team: np.ndarray) -> np.ndarray:
    """Assign a cluster id to each frame, one per unbroken possession run.

    This is the resampling unit for the cluster bootstrap. Windows inside one
    possession are near-duplicates of each other; resampling them independently
    would treat correlated observations as independent evidence and produce
    confidence intervals far narrower than the data supports.

    Args:
        possession_team: Possessing team per frame.

    Returns:
        Integer cluster id per frame.
    """
    changes = np.zeros(possession_team.shape[0], dtype=bool)
    changes[1:] = possession_team[1:] != possession_team[:-1]
    return np.cumsum(changes)


def build_labels(
    *,
    match_id: str,
    ball_xy: np.ndarray,
    times_s: np.ndarray,
    periods: np.ndarray,
    possession_team: np.ndarray,
    dead_ball: np.ndarray,
    orientation: Orientation,
    window: WindowSettings,
    episode_settings: EpisodeSettings,
    frame_rate: float,
) -> LabelledWindows:
    """Construct prediction samples and their labels.

    A sample is emitted every ``stride_s`` at instants where a prediction would
    genuinely be made. Four kinds of instant are excluded, each for a reason:

    * too early in the period to have a full observation window;
    * the ball is out of play, so there is nothing to predict;
    * possession is unknown, so there is no attacking team to predict for;
    * the ball is already inside the penalty area, so the event being predicted
      has already happened.

    Labels look strictly forward from the prediction instant and features
    strictly backward, so the two never overlap.

    Args:
        match_id: Match identifier.
        ball_xy: Ball positions in canonical coordinates.
        times_s: Match time per frame.
        periods: Period per frame.
        possession_team: Possessing team per frame, ``0`` home, ``1`` away, ``-1``
            unknown, derived causally.
        dead_ball: Mask of frames with play stopped.
        orientation: Attacking direction per period and team.
        window: Window configuration.
        episode_settings: Episode grouping configuration.
        frame_rate: Tracking sample rate in hertz.

    Returns:
        The labelled samples.
    """
    entries = find_entries(ball_xy, times_s, periods, possession_team, orientation)
    episodes = merge_episodes(entries, match_id, periods, episode_settings.merge_gap_s)
    clusters = possession_clusters(possession_team)

    observation_frames = round(window.observation_s * frame_rate)
    stride_frames = max(1, round(window.stride_s * frame_rate))
    horizon_s = window.horizon_s

    # Ball-in-box state per frame, for the "already in the box" exclusion.
    in_box = np.zeros(ball_xy.shape[0], dtype=bool)
    for period in np.unique(periods):
        for code, team in ((0, Team.HOME), (1, Team.AWAY)):
            mask = (periods == period) & (possession_team == code)
            if mask.any():
                sign = orientation.direction(int(period), team).sign
                in_box[mask] = box_entry_mask(ball_xy[mask], sign)

    episode_times = np.array([e.entry_time_s for e in episodes])
    episode_teams = np.array([0 if e.team is Team.HOME else 1 for e in episodes])

    ends: list[int] = []
    labels: list[int] = []
    episode_ids: list[int] = []

    # Only sample instants that have a full window of history *within the same
    # period*; a window straddling half time would mix playing directions.
    for period in np.unique(periods):
        idx = np.flatnonzero(periods == period)
        if idx.size <= observation_frames:
            continue
        for i in idx[observation_frames::stride_frames]:
            if dead_ball[i] or possession_team[i] < 0 or in_box[i]:
                continue
            now = times_s[i]
            team_code = possession_team[i]
            # Strictly forward: an entry exactly at `now` belongs to the past.
            future = (
                (episode_times > now)
                & (episode_times <= now + horizon_s)
                & (episode_teams == team_code)
            )
            hit = np.flatnonzero(future)
            ends.append(int(i))
            labels.append(1 if hit.size else 0)
            episode_ids.append(int(hit[0]) if hit.size else -1)

    end_index = np.array(ends, dtype=np.int64)
    return LabelledWindows(
        match_id=match_id,
        end_index=end_index,
        time_s=times_s[end_index] if end_index.size else np.array([]),
        period=periods[end_index] if end_index.size else np.array([]),
        attacking_team=possession_team[end_index] if end_index.size else np.array([]),
        label=np.array(labels, dtype=np.int8),
        episode_id=np.array(episode_ids, dtype=np.int64),
        cluster_id=clusters[end_index] if end_index.size else np.array([]),
        episodes=episodes,
    )
