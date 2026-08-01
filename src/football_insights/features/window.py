"""Rolling observation windows.

The live path and the training path must produce byte-identical inputs for the
same instant, otherwise evaluation measures something the service never sees.
Two details do the work:

**Velocity warm-up.** Backward-difference velocity needs history before the
first retained frame. The buffer therefore holds ``observation + span`` frames
and discards the leading ``span`` rows after computing features, so every
retained row has real history rather than the edge-padded fallback. Offline
extraction slices the same way.

**Shared subsampling.** Both paths reduce the raw window to the model's
sequence length through :func:`subsample_indices`, so there is exactly one rule
for which frames a model sees.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from football_insights.features.frame_features import (
    VELOCITY_SPAN_S,
    PossessionContext,
    compute_features,
)
from football_insights.features.spec import DEFAULT_FEATURE_SPEC, FeatureSpec
from football_insights.pitch import DEFAULT_PITCH, Pitch

if TYPE_CHECKING:
    from football_insights.config import WindowSettings
    from football_insights.domain import Frame


def subsample_indices(n_raw: int, sequence_length: int) -> np.ndarray:
    """Row indices reducing a raw window to the model's sequence length.

    Always includes the final (most recent) row, because that is the instant the
    prediction is made for.

    Args:
        n_raw: Number of raw frames in the window.
        sequence_length: Number of timesteps the model expects.

    Returns:
        Integer indices of length ``sequence_length``, ascending.
    """
    if n_raw <= 0:
        msg = "cannot subsample an empty window"
        raise ValueError(msg)
    if sequence_length <= 1:
        return np.array([n_raw - 1], dtype=np.int64)
    return np.rint(np.linspace(0, n_raw - 1, sequence_length)).astype(np.int64)


@dataclass(frozen=True, slots=True)
class WindowValidity:
    """Why a window is or is not usable.

    A window that is not ``ok`` must never produce an insight. The reason is
    carried through to the suppression metrics so operators can tell a data
    problem apart from a quiet model.
    """

    ok: bool
    reason: str
    valid_frame_ratio: float
    n_frames: int

    @classmethod
    def valid(cls, ratio: float, n: int) -> WindowValidity:
        """A usable window."""
        return cls(True, "ok", ratio, n)


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    """Frame counts derived from the window settings and the tracking rate."""

    observation_frames: int
    warmup_frames: int
    sequence_length: int

    @property
    def capacity(self) -> int:
        """Total frames the buffer must hold."""
        return self.observation_frames + self.warmup_frames

    @classmethod
    def build(cls, settings: WindowSettings, frame_rate: float) -> WindowGeometry:
        """Derive geometry from configuration.

        Args:
            settings: Window configuration.
            frame_rate: Tracking sample rate in hertz.

        Returns:
            The derived geometry.
        """
        return cls(
            observation_frames=max(1, round(settings.observation_s * frame_rate)),
            warmup_frames=max(1, round(VELOCITY_SPAN_S * frame_rate)),
            sequence_length=settings.sequence_length,
        )


class RollingWindow:
    """Fixed-capacity buffer of the most recent frames, for live inference.

    The buffer is incremental: appending a frame is O(1) and never re-reads the
    match. Feature computation happens only when a window is extracted.

    Args:
        geometry: Frame counts for this window.
        min_valid_frame_ratio: Fraction of frames that must carry a finite ball
            position for the window to be considered structurally valid.
        spec: Feature schema.
        pitch: Pitch dimensions.
    """

    __slots__ = ("_frames", "_geometry", "_min_ratio", "_pitch", "_spec")

    def __init__(
        self,
        geometry: WindowGeometry,
        min_valid_frame_ratio: float,
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
        pitch: Pitch = DEFAULT_PITCH,
    ) -> None:
        """Create an empty buffer."""
        self._geometry = geometry
        self._min_ratio = min_valid_frame_ratio
        self._spec = spec
        self._pitch = pitch
        self._frames: deque[Frame] = deque(maxlen=geometry.capacity)

    def __len__(self) -> int:
        """Number of frames currently buffered."""
        return len(self._frames)

    @property
    def geometry(self) -> WindowGeometry:
        """The window geometry in use."""
        return self._geometry

    @property
    def is_full(self) -> bool:
        """Whether the buffer holds a complete window."""
        return len(self._frames) == self._geometry.capacity

    @property
    def latest(self) -> Frame | None:
        """The most recently appended frame."""
        return self._frames[-1] if self._frames else None

    def clear(self) -> None:
        """Drop all buffered frames, used after a replay interruption."""
        self._frames.clear()

    def append(self, frame: Frame) -> None:
        """Add a frame, evicting the oldest when at capacity.

        Frames arriving out of order or duplicating the current head are
        rejected here rather than silently corrupting the window; the replay
        layer counts them.

        Args:
            frame: The frame to add.

        Raises:
            ValueError: If the frame is not strictly newer than the buffer head.
        """
        head = self.latest
        if head is not None and frame.frame <= head.frame:
            msg = (
                f"frame {frame.frame} is not newer than buffered head {head.frame}; "
                "the replay layer must reorder or drop it"
            )
            raise ValueError(msg)
        if head is not None and frame.period != head.period:
            # A period change invalidates velocity history and playing direction.
            self._frames.clear()
        self._frames.append(frame)

    def validity(self) -> WindowValidity:
        """Assess whether the buffered window can be scored."""
        n = len(self._frames)
        if n < self._geometry.capacity:
            return WindowValidity(False, "insufficient_frames", 0.0, n)
        ball = np.array([f.ball_xy for f in self._frames])
        finite = np.isfinite(ball).all(axis=1)
        ratio = float(finite.mean())
        if ratio < self._min_ratio:
            return WindowValidity(False, "invalid_window", ratio, n)
        if not finite[-1]:
            # The instant being predicted for must itself be observed.
            return WindowValidity(False, "invalid_window", ratio, n)
        return WindowValidity.valid(ratio, n)

    def stacked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stack the buffer into arrays.

        Returns:
            Tuple of ``(period, frame, time_s, home_xy, away_xy)`` plus ball is
            returned separately by :meth:`ball`.
        """
        return (
            np.array([f.period for f in self._frames]),
            np.array([f.frame for f in self._frames]),
            np.array([f.time_s for f in self._frames]),
            np.stack([f.home_xy for f in self._frames]),
            np.stack([f.away_xy for f in self._frames]),
        )

    def ball(self) -> np.ndarray:
        """Ball positions for the buffered frames, shape ``(capacity, 2)``."""
        return np.stack([f.ball_xy for f in self._frames])


def extract_window(
    features: np.ndarray,
    end_index: int,
    geometry: WindowGeometry,
) -> np.ndarray:
    """Slice a model input from a full-match feature matrix.

    Args:
        features: Feature matrix for the whole match, ``(n_frames, n_features)``.
        end_index: Row index of the prediction instant, inclusive.
        geometry: Window geometry.

    Returns:
        Array ``(sequence_length, n_features)``.

    Raises:
        IndexError: If there is not enough history before ``end_index``.
    """
    start = end_index - geometry.observation_frames + 1
    if start < 0:
        msg = (
            f"window ending at row {end_index} needs {geometry.observation_frames} "
            f"frames of history but only {end_index + 1} are available"
        )
        raise IndexError(msg)
    raw = features[start : end_index + 1]
    return np.asarray(raw[subsample_indices(raw.shape[0], geometry.sequence_length)])


def window_features_from_buffer(
    buffer: RollingWindow,
    *,
    direction_sign: float,
    frame_rate: float,
    possession: PossessionContext,
    attack_is_home: bool,
    attack_is_gk: np.ndarray | None = None,
    defend_is_gk: np.ndarray | None = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    pitch: Pitch = DEFAULT_PITCH,
) -> np.ndarray:
    """Compute a model input from the live buffer.

    Calls the same :func:`~football_insights.features.frame_features.compute_features`
    the training path uses, then discards the velocity warm-up rows so the
    result matches an offline slice of the same instant exactly.

    Args:
        buffer: The rolling buffer, which must be full.
        direction_sign: ``+1`` or ``-1`` for the attacking team.
        frame_rate: Tracking sample rate in hertz.
        possession: Causal possession context covering the buffered frames.
        attack_is_home: Whether the home team is the attacking team.
        attack_is_gk: Goalkeeper mask for the attacking team's columns.
        defend_is_gk: Goalkeeper mask for the defending team's columns.
        spec: Feature schema.
        pitch: Pitch dimensions.

    Returns:
        Array ``(sequence_length, n_features)``.

    Raises:
        ValueError: If the buffer is not yet full.
    """
    if not buffer.is_full:
        msg = f"buffer holds {len(buffer)} frames, needs {buffer.geometry.capacity}"
        raise ValueError(msg)

    _, _, _, home_xy, away_xy = buffer.stacked()
    ball = buffer.ball()
    attack_xy = home_xy if attack_is_home else away_xy
    defend_xy = away_xy if attack_is_home else home_xy

    features = compute_features(
        attack_xy=attack_xy,
        defend_xy=defend_xy,
        ball_xy=ball,
        direction_sign=direction_sign,
        frame_rate=frame_rate,
        possession=possession,
        attack_is_gk=attack_is_gk,
        defend_is_gk=defend_is_gk,
        pitch=pitch,
        spec=spec,
    )
    retained = features[buffer.geometry.warmup_frames :]
    return np.asarray(
        retained[subsample_indices(retained.shape[0], buffer.geometry.sequence_length)]
    )
