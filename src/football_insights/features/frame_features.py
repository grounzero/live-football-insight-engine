"""Per-frame feature computation.

A single vectorised implementation serves both offline training and live
inference. That is deliberate: two implementations of "the same" features is the
classic source of train/serve skew, where a model performs well in evaluation
and quietly degrades in production. Here the live path calls exactly the
function the training path calls, over a shorter array.

Two properties make that possible:

**Causal derivatives.** Velocities use backward differences only. Central
differences would make a timestep's features depend on frames after it, so the
same instant would produce different values depending on whether it sat in the
middle of a window (offline) or at its end (live). Backward differences are
identical in both cases and are genuinely causal.

**Orientation.** Coordinates are rotated 180 degrees when the team in possession
attacks -x, so every feature is expressed as "toward the goal being attacked".
Rotation rather than mirroring preserves the handedness of lateral features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.pitch import DEFAULT_PITCH, Pitch

if TYPE_CHECKING:
    from football_insights.features.causal import CausalEventView

#: Velocity is measured over this many seconds of history. Raw 25 Hz tracking is
#: noisy frame to frame; a short backward span damps that without lag.
VELOCITY_SPAN_S: Final = 0.12

#: Radius used by the "defenders close to the ball" count.
PRESSURE_RADIUS_M: Final = 10.0

#: Half-angle of the cone ahead of the ball used by ``space_ahead_of_ball``.
SPACE_CONE_HALF_ANGLE_RAD: Final = np.pi / 4.0


@dataclass(frozen=True, slots=True)
class PossessionContext:
    """Per-frame possession context, already derived causally.

    Every array has length ``T`` and is produced by
    :func:`build_possession_context`, which reads only a
    :class:`~football_insights.features.causal.CausalEventView`.
    """

    duration_s: np.ndarray
    event_count: np.ndarray
    is_dead_ball: np.ndarray
    event_in_flight: np.ndarray
    recent_pass_count: np.ndarray
    recent_box_entry_count: np.ndarray
    time_since_last_box_entry: np.ndarray


def _backward_velocity(xy: np.ndarray, span: int, dt: float) -> np.ndarray:
    """Backward-difference velocity in metres per second.

    Args:
        xy: Positions shaped ``(T, ..., 2)``.
        span: Number of frames to look back.
        dt: Seconds per frame.

    Returns:
        Velocities of the same shape. The first ``span`` rows reuse the earliest
        computable value rather than reporting a spurious zero.
    """
    if xy.shape[0] <= span:
        return np.zeros_like(xy)
    delta = np.empty_like(xy)
    delta[span:] = (xy[span:] - xy[:-span]) / (span * dt)
    delta[:span] = delta[span]
    return delta


def _sorted_distances(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame distances from every player to a per-frame target, sorted ascending.

    Args:
        points: Player positions shaped ``(T, P, 2)``.
        target: Target positions shaped ``(T, 2)``.

    Returns:
        Array ``(T, P)`` of distances with ``NaN`` (absent players) sorted last.
    """
    delta = points - target[:, None, :]
    dist = np.hypot(delta[..., 0], delta[..., 1])
    return np.sort(dist, axis=1)


def _nan_safe(values: np.ndarray, fill: float) -> np.ndarray:
    """Replace non-finite entries with a fixed value."""
    return np.where(np.isfinite(values), values, fill)


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmean`` that tolerates all-NaN slices.

    A frame where every player of a team is absent is a data problem for the
    window validator to judge, not a reason to emit a warning on the hot path.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmean(values, axis=axis))


def _nanstd(values: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanstd`` that tolerates all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanstd(values, axis=axis))


def _nanmax(values: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmax`` that tolerates all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmax(values, axis=axis))


def box_entry_history(
    in_box: np.ndarray,
    times_s: np.ndarray,
    lookback_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal history of penalty-area entries.

    An entry is a rising edge of ``in_box``. Both outputs at index ``i`` use
    only entries that occurred strictly before frame ``i``, so a frame never
    counts its own entry — that would leak the very thing being predicted.

    Args:
        in_box: Boolean mask of frames with the ball inside the attacking box.
        times_s: Match time per frame, shape ``(T,)``.
        lookback_s: Width of the counting window.

    Returns:
        Tuple of ``(count_within_lookback, seconds_since_last_entry)``. When no
        entry has yet occurred the recency is saturated at ``lookback_s``.
    """
    n = in_box.shape[0]
    rising = np.zeros(n, dtype=bool)
    if n > 1:
        rising[1:] = in_box[1:] & ~in_box[:-1]
    entry_times = times_s[rising]

    counts = np.zeros(n)
    since = np.full(n, lookback_s)
    if entry_times.size:
        # searchsorted with side="left" keeps the comparison strict, so an entry
        # at exactly this frame is not visible to it.
        idx = np.searchsorted(entry_times, times_s, side="left")
        lower = np.searchsorted(entry_times, times_s - lookback_s, side="left")
        counts = (idx - lower).astype(float)
        has_prev = idx > 0
        last = np.where(has_prev, entry_times[np.clip(idx - 1, 0, entry_times.size - 1)], np.nan)
        since = np.where(has_prev, np.minimum(times_s - last, lookback_s), lookback_s)
    return counts, since


def build_possession_context(
    view: CausalEventView,
    frames: np.ndarray,
    times_s: np.ndarray,
    in_box: np.ndarray,
    frame_rate: float,
    recent_window_s: float = 10.0,
    box_entry_window_s: float = 120.0,
) -> PossessionContext:
    """Derive per-frame possession context from the causal event view.

    Args:
        view: Forward-blind event access.
        frames: Source frame indices, shape ``(T,)``.
        times_s: Match time per frame, shape ``(T,)``.
        in_box: Mask of frames with the ball inside the attacking penalty area,
            used for the causal entry history.
        frame_rate: Tracking sample rate in hertz.
        recent_window_s: Lookback for the recent pass count.
        box_entry_window_s: Lookback for recent penalty-area entries.

    Returns:
        Context arrays aligned with ``frames``.
    """
    from football_insights.domain import EventType

    n = frames.shape[0]
    duration = np.zeros(n)
    count = np.zeros(n)
    dead = np.zeros(n)
    in_flight = np.zeros(n)
    passes = np.zeros(n)
    recent_frames = int(recent_window_s * frame_rate)

    for i in range(n):
        now = int(frames[i])
        state = view.possession(now)
        duration[i] = state.duration_s
        count[i] = state.event_count
        dead[i] = float(state.is_dead_ball)
        in_flight[i] = float(state.has_event_in_flight)
        counts = view.recent_type_counts(now, recent_frames)
        passes[i] = counts.get(EventType.PASS, 0) + counts.get(EventType.CARRY, 0)

    entries, since_entry = box_entry_history(in_box, times_s, box_entry_window_s)

    return PossessionContext(
        duration_s=duration,
        event_count=count,
        is_dead_ball=dead,
        event_in_flight=in_flight,
        recent_pass_count=passes,
        recent_box_entry_count=entries,
        time_since_last_box_entry=since_entry,
    )


def compute_features(
    *,
    attack_xy: np.ndarray,
    defend_xy: np.ndarray,
    ball_xy: np.ndarray,
    direction_sign: float,
    frame_rate: float,
    possession: PossessionContext,
    attack_is_gk: np.ndarray | None = None,
    defend_is_gk: np.ndarray | None = None,
    pitch: Pitch = DEFAULT_PITCH,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> np.ndarray:
    """Compute the full feature matrix for a run of frames.

    Args:
        attack_xy: Positions of the team in possession, ``(T, P, 2)`` canonical.
        defend_xy: Positions of the defending team, ``(T, Q, 2)`` canonical.
        ball_xy: Ball positions, ``(T, 2)`` canonical.
        direction_sign: ``+1`` if the attacking team attacks ``+x`` in the
            canonical frame, ``-1`` otherwise.
        frame_rate: Tracking sample rate in hertz.
        possession: Causally derived possession context of length ``T``.
        attack_is_gk: Boolean mask of goalkeeper columns in ``attack_xy``.
        defend_is_gk: Boolean mask of goalkeeper columns in ``defend_xy``.
        pitch: Pitch dimensions.
        spec: Feature schema; controls column order of the result.

    Returns:
        Array of shape ``(T, spec.n_features)`` in ``float32``.
    """
    # Rotate so the attacking team always attacks +x.
    atk = attack_xy * direction_sign
    dfd = defend_xy * direction_sign
    ball = ball_xy * direction_sign

    n = ball.shape[0]
    dt = 1.0 / frame_rate
    span = max(1, round(VELOCITY_SPAN_S * frame_rate))

    ball_v = _backward_velocity(ball, span, dt)
    atk_v = _backward_velocity(atk, span, dt)
    dfd_v = _backward_velocity(dfd, span, dt)

    goal_x = pitch.half_length
    final_third_x = pitch.half_length / 3.0

    cols: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- ball
    cols["ball_x"] = ball[:, 0] / pitch.half_length
    cols["ball_y_abs"] = np.abs(ball[:, 1]) / pitch.half_width
    cols["ball_speed"] = np.hypot(ball_v[:, 0], ball_v[:, 1])
    cols["ball_v_goalward"] = ball_v[:, 0]
    cols["ball_v_lateral"] = np.abs(ball_v[:, 1])
    cols["ball_dist_to_goal"] = pitch.distance_to_attacking_goal(ball)
    cols["ball_goal_angle"] = pitch.goal_visible_angle(ball)
    cols["ball_in_final_third"] = (ball[:, 0] >= final_third_x).astype(np.float64)
    ball_in_box = pitch.is_inside_penalty_area(ball)
    cols["ball_in_box"] = ball_in_box.astype(np.float64)

    # ------------------------------------------------------- attacking shape
    # The attacking goalkeeper sits far behind play and would drag the centroid,
    # depth and width of the attacking block backwards, so shape aggregates use
    # outfielders only. Counts (players ahead of the ball, in the box) are
    # unaffected because a keeper is never in those regions when attacking.
    atk_x_all = atk[..., 0]
    atk_outfield = atk_x_all.copy()
    atk_y_outfield = atk[..., 1].copy()
    if attack_is_gk is not None and attack_is_gk.any():
        atk_outfield[:, attack_is_gk] = np.nan
        atk_y_outfield[:, attack_is_gk] = np.nan
    atk_x = atk_x_all
    ahead = atk_x > ball[:, 0][:, None]
    cols["attackers_ahead_of_ball"] = np.nansum(ahead & np.isfinite(atk_x), axis=1).astype(float)
    cols["attackers_in_box"] = pitch.is_inside_penalty_area(atk).sum(axis=1).astype(float)
    cols["attackers_in_final_third"] = np.nansum(
        (atk_x >= final_third_x) & np.isfinite(atk_x), axis=1
    ).astype(float)

    with np.errstate(invalid="ignore"):
        cols["attack_centroid_x"] = _nan_safe(_nanmean(atk_outfield, 1), 0.0)
        cols["attack_width"] = _nan_safe(_nanstd(atk_y_outfield, 1), 0.0)
        cols["attack_depth"] = _nan_safe(_nanstd(atk_outfield, 1), 0.0)
        cols["attack_highest_x"] = _nan_safe(_nanmax(atk_outfield, 1), 0.0)
        cols["attack_mean_v_goalward"] = _nan_safe(_nanmean(atk_v[..., 0], 1), 0.0)
    cols["attackers_moving_goalward"] = np.nansum(
        (atk_v[..., 0] > 0.5) & np.isfinite(atk_v[..., 0]), axis=1
    ).astype(float)

    # ------------------------------------------------------- defending shape
    dfd_x = dfd[..., 0]
    dfd_y = dfd[..., 1]
    cols["defenders_in_box"] = pitch.is_inside_penalty_area(dfd).sum(axis=1).astype(float)
    goalside = dfd_x > ball[:, 0][:, None]
    cols["defenders_goalside_of_ball"] = np.nansum(goalside & np.isfinite(dfd_x), axis=1).astype(
        float
    )

    # Defensive line: mean x of the four outfielders deepest toward their own
    # goal, excluding the keeper so a sweeper-keeper does not drag the line.
    outfield = dfd_x.copy()
    if defend_is_gk is not None and defend_is_gk.any():
        outfield[:, defend_is_gk] = np.nan
    sorted_def_x = np.sort(np.where(np.isfinite(outfield), outfield, np.inf), axis=1)
    line = sorted_def_x[:, :4]
    line = np.where(np.isfinite(line), line, np.nan)
    with np.errstate(invalid="ignore"):
        cols["defensive_line_x"] = _nan_safe(_nanmean(line, 1), 0.0)
        cols["defence_centroid_x"] = _nan_safe(_nanmean(outfield, 1), 0.0)
        centroid_y = _nan_safe(_nanmean(dfd_y, 1), 0.0)
        spread = np.hypot(
            outfield - cols["defence_centroid_x"][:, None], dfd_y - centroid_y[:, None]
        )
        cols["defensive_compactness"] = _nan_safe(_nanmean(spread, 1), 0.0)
        cols["defence_mean_v_goalward"] = _nan_safe(_nanmean(dfd_v[..., 0], 1), 0.0)

    if defend_is_gk is not None and defend_is_gk.any():
        gk_xy = dfd[:, defend_is_gk, :][:, 0, :]
    else:
        # Fall back to the deepest defender when no keeper is identified.
        deepest = np.nanargmin(np.where(np.isfinite(dfd_x), dfd_x, np.inf), axis=1)
        gk_xy = dfd[np.arange(n), deepest, :]
    cols["gk_dist_to_ball"] = _nan_safe(
        np.hypot(gk_xy[:, 0] - ball[:, 0], gk_xy[:, 1] - ball[:, 1]), pitch.length
    )
    cols["gk_dist_to_goal"] = _nan_safe(pitch.distance_to_attacking_goal(gk_xy), pitch.length)

    # ---------------------------------------------------------- ball contest
    def_dist = _sorted_distances(dfd, ball)
    atk_dist = _sorted_distances(atk, ball)
    far = pitch.length
    cols["nearest_defender_dist"] = _nan_safe(def_dist[:, 0], far)
    cols["second_nearest_defender_dist"] = _nan_safe(
        def_dist[:, 1] if def_dist.shape[1] > 1 else def_dist[:, 0], far
    )
    with np.errstate(invalid="ignore"):
        cols["mean_three_nearest_defender_dist"] = _nan_safe(
            _nanmean(def_dist[:, : min(3, def_dist.shape[1])], 1), far
        )
    # The nearest attacker to the ball is usually the carrier, so support is the
    # second nearest.
    cols["nearest_attacker_support_dist"] = _nan_safe(
        atk_dist[:, 1] if atk_dist.shape[1] > 1 else atk_dist[:, 0], far
    )
    cols["defenders_within_10m_of_ball"] = np.nansum(def_dist <= PRESSURE_RADIUS_M, axis=1).astype(
        float
    )

    # Space ahead: distance to the nearest defender inside a forward cone from
    # the ball toward the goal. Large values mean a clear route to run into.
    rel = dfd - ball[:, None, :]
    ahead_mask = rel[..., 0] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.abs(np.arctan2(rel[..., 1], np.maximum(rel[..., 0], 1e-6)))
    cone = ahead_mask & (angle <= SPACE_CONE_HALF_ANGLE_RAD) & np.isfinite(rel[..., 0])
    cone_dist = np.where(cone, np.hypot(rel[..., 0], rel[..., 1]), np.inf)
    space = cone_dist.min(axis=1)
    goal_gap = np.maximum(goal_x - ball[:, 0], 0.0)
    cols["space_ahead_of_ball"] = np.minimum(_nan_safe(space, far), goal_gap)

    # ------------------------------------------------------------ possession
    cols["possession_duration_s"] = possession.duration_s
    cols["possession_event_count"] = possession.event_count
    cols["is_dead_ball"] = possession.is_dead_ball
    cols["event_in_flight"] = possession.event_in_flight
    cols["recent_pass_count"] = possession.recent_pass_count
    cols["recent_box_entry_count"] = possession.recent_box_entry_count
    cols["time_since_last_box_entry"] = possession.time_since_last_box_entry

    missing = set(spec.names) - set(cols)
    if missing:
        msg = f"feature spec expects columns that were not computed: {sorted(missing)}"
        raise KeyError(msg)
    extra = set(cols) - set(spec.names)
    if extra:
        msg = f"computed columns absent from the feature spec: {sorted(extra)}"
        raise KeyError(msg)

    out = np.empty((n, spec.n_features), dtype=np.float32)
    for j, name in enumerate(spec.names):
        out[:, j] = cols[name]
    # Absent players and edge frames can still leave a NaN; the window validator
    # decides whether a window is usable, but the array itself must be finite so
    # a model never receives NaN.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def box_entry_mask(
    ball_xy: np.ndarray, direction_sign: float, pitch: Pitch = DEFAULT_PITCH
) -> np.ndarray:
    """Boolean mask of frames where the ball is inside the attacking penalty area.

    Args:
        ball_xy: Ball positions ``(T, 2)`` in canonical coordinates.
        direction_sign: ``+1`` or ``-1`` for the attacking team.
        pitch: Pitch dimensions.

    Returns:
        Boolean array ``(T,)``.
    """
    return pitch.is_inside_penalty_area(ball_xy * direction_sign)
