"""Temporal-leakage safeguards.

These are the tests the whole project's credibility rests on. If features
computed at time ``t`` can be changed by editing events after ``t``, every
reported metric is optimistic and the live system would behave differently from
evaluation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pytest

from football_insights.data.synthetic import SyntheticMatch, generate_synthetic_match
from football_insights.domain import Event, EventType, Team
from football_insights.features.causal import CausalEventView
from football_insights.features.frame_features import (
    box_entry_history,
    box_entry_mask,
    build_possession_context,
    compute_features,
)
from tests.support import approx


@pytest.fixture(scope="module")
def match() -> SyntheticMatch:
    return generate_synthetic_match(seed=11, period_duration_s=90.0)


def _features_at(match: SyntheticMatch, events: Sequence[Event], upto_frame: int) -> np.ndarray:
    """Build the feature matrix for period 1 frames up to ``upto_frame``."""
    t = match.tracking
    mask = (t.period == 1) & (t.frame <= upto_frame)
    idx = np.flatnonzero(mask)
    view = CausalEventView(events, t.frame_rate)
    sign = match.orientation.direction(1, Team.HOME).sign
    ball = t.ball_xy[idx]
    in_box = box_entry_mask(ball, sign)
    pc = build_possession_context(view, t.frame[idx], t.time_s[idx], in_box, t.frame_rate)
    gk_a = np.array([p.is_goalkeeper for p in t.players(Team.HOME)])
    gk_d = np.array([p.is_goalkeeper for p in t.players(Team.AWAY)])
    return compute_features(
        attack_xy=t.team_xy(Team.HOME)[idx],
        defend_xy=t.team_xy(Team.AWAY)[idx],
        ball_xy=ball,
        direction_sign=sign,
        frame_rate=t.frame_rate,
        possession=pc,
        attack_is_gk=gk_a,
        defend_is_gk=gk_d,
    )


class TestFutureEventsAreInvisible:
    """Rule 1: an event does not exist until it has started."""

    def test_deleting_future_events_changes_nothing(self, match: SyntheticMatch) -> None:
        cut = int(match.tracking.frame[len(match.tracking.frame) // 3])
        base = _features_at(match, match.events, cut)
        pruned = tuple(e for e in match.events if e.start_frame <= cut)
        assert len(pruned) < len(match.events), "fixture must have events after the cut"
        np.testing.assert_array_equal(base, _features_at(match, pruned, cut))

    def test_mutating_future_events_changes_nothing(self, match: SyntheticMatch) -> None:
        cut = int(match.tracking.frame[len(match.tracking.frame) // 3])
        base = _features_at(match, match.events, cut)
        mutated = tuple(
            e
            if e.start_frame <= cut
            else dataclasses.replace(
                e,
                team=e.team.opponent,
                type=EventType.SHOT,
                start_xy=(40.0, 5.0),
                end_xy=(52.0, 0.0),
                to_player="tampered",
            )
            for e in match.events
        )
        np.testing.assert_array_equal(base, _features_at(match, mutated, cut))

    def test_fabricating_future_events_changes_nothing(self, match: SyntheticMatch) -> None:
        cut = int(match.tracking.frame[len(match.tracking.frame) // 3])
        base = _features_at(match, match.events, cut)
        invented = tuple(match.events) + tuple(
            Event(
                team=Team.AWAY,
                type=EventType.SHOT,
                subtype="ON TARGET-GOAL",
                period=1,
                start_frame=cut + offset,
                end_frame=cut + offset + 5,
                start_time_s=(cut + offset) / match.frame_rate,
                end_time_s=(cut + offset + 5) / match.frame_rate,
                start_xy=(45.0, 0.0),
                end_xy=(52.5, 0.0),
            )
            for offset in (1, 10, 200)
        )
        np.testing.assert_array_equal(base, _features_at(match, invented, cut))

    @pytest.mark.parametrize("fraction", [0.2, 0.45, 0.7, 0.95])
    def test_property_holds_across_the_match(self, match: SyntheticMatch, fraction: float) -> None:
        frames = match.tracking.frame[match.tracking.period == 1]
        cut = int(frames[int(len(frames) * fraction)])
        base = _features_at(match, match.events, cut)
        pruned = tuple(e for e in match.events if e.start_frame <= cut)
        np.testing.assert_array_equal(base, _features_at(match, pruned, cut))


class TestInFlightEventsHideTheirOutcome:
    """Rule 2: a visible event's outcome stays hidden until it resolves."""

    def test_outcome_fields_are_none_while_in_flight(self) -> None:
        ev = Event(
            team=Team.HOME,
            type=EventType.PASS,
            subtype=None,
            period=1,
            start_frame=100,
            end_frame=140,
            start_time_s=4.0,
            end_time_s=5.6,
            from_player="home_7",
            to_player="home_9",
            start_xy=(0.0, 0.0),
            end_xy=(30.0, 10.0),
        )
        view = CausalEventView([ev], 25.0)

        mid = view.latest(120)
        assert mid is not None
        assert mid.resolved is False
        assert mid.to_player is None, "recipient reveals that the pass completes"
        assert mid.end_xy is None, "end location reveals where the ball arrives"
        assert mid.end_frame is None
        assert mid.type is EventType.PASS and mid.from_player == "home_7"

        done = view.latest(140)
        assert done is not None
        assert done.resolved is True
        assert done.to_player == "home_9"
        assert done.end_xy == (30.0, 10.0)

    def test_mutating_an_inflight_outcome_changes_nothing(self, match: SyntheticMatch) -> None:
        t = match.tracking
        straddling = [e for e in match.events if e.end_frame > e.start_frame + 2 and e.period == 1]
        assert straddling, "fixture must contain multi-frame events"
        target = straddling[len(straddling) // 2]
        cut = target.start_frame + 1
        assert target.end_frame > cut

        base = _features_at(match, match.events, cut)
        tampered = tuple(
            dataclasses.replace(e, end_xy=(52.0, 0.0), to_player="ghost", end_time_s=e.end_time_s)
            if e is target
            else e
            for e in match.events
        )
        np.testing.assert_array_equal(base, _features_at(match, tampered, cut))
        assert t.n_frames > 0


class TestPossessionIsCausal:
    def test_duration_measures_to_now_not_to_sequence_end(self) -> None:
        events = [
            Event(
                team=Team.HOME,
                type=EventType.PASS,
                subtype=None,
                period=1,
                start_frame=100,
                end_frame=130,
                start_time_s=4.0,
                end_time_s=5.2,
            ),
            Event(
                team=Team.HOME,
                type=EventType.PASS,
                subtype=None,
                period=1,
                start_frame=130,
                end_frame=400,
                start_time_s=5.2,
                end_time_s=16.0,
            ),
        ]
        view = CausalEventView(events, 25.0)
        state = view.possession(150)
        assert state.team is Team.HOME
        # 50 frames after possession began at frame 100, at 25 Hz.
        assert state.duration_s == approx(2.0)
        assert state.event_count == 2
        assert state.has_event_in_flight is True

    def test_possession_before_any_event_is_unknown(self) -> None:
        view = CausalEventView([], 25.0)
        state = view.possession(10)
        assert state.team is None
        assert state.duration_s == 0.0

    def test_turnover_resets_the_run(self) -> None:
        events = [
            Event(Team.HOME, EventType.PASS, None, 1, 10, 20, 0.4, 0.8),
            Event(Team.AWAY, EventType.RECOVERY, None, 1, 30, 30, 1.2, 1.2),
        ]
        view = CausalEventView(events, 25.0)
        assert view.possession(25).team is Team.HOME
        after = view.possession(35)
        assert after.team is Team.AWAY
        assert after.event_count == 1


class TestBoxEntryHistoryIsCausal:
    def test_a_frame_never_counts_its_own_entry(self) -> None:
        times = np.arange(0, 10, 1.0)
        in_box = np.zeros(10, dtype=bool)
        in_box[5:8] = True  # rising edge at index 5
        counts, since = box_entry_history(in_box, times, lookback_s=60.0)
        assert counts[5] == 0, "the entering frame must not see its own entry"
        assert counts[6] == 1
        assert since[5] == 60.0
        assert since[6] == approx(1.0)

    def test_counts_respect_the_lookback(self) -> None:
        times = np.arange(0, 200, 1.0)
        in_box = np.zeros(200, dtype=bool)
        in_box[10] = True
        in_box[150] = True
        counts, _ = box_entry_history(in_box, times, lookback_s=60.0)
        assert counts[100] == 0, "an entry 90 s ago is outside a 60 s lookback"
        assert counts[160] == 1
