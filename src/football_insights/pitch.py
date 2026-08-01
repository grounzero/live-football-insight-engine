"""Pitch geometry and coordinate conventions.

Two coordinate systems are used, and the boundary between them is deliberate:

**Source coordinates** (Metrica): ``x, y in [0, 1]``, ``(0, 0)`` top-left,
``(1, 1)`` bottom-right, ``(0.5, 0.5)`` the centre spot.

**Canonical coordinates** (everything downstream): metres, origin at the centre
spot, ``+x`` toward the right-hand goal, ``+y`` toward the top touchline as
drawn. The pitch therefore spans ``x in [-52.5, 52.5]``, ``y in [-34, 34]``.

Reorienting so that the team in possession always attacks ``+x`` is a **180 degree
rotation** (negate both x and y), not a mirror of x alone. Mirroring would swap
left and right wings and silently corrupt any handedness-sensitive feature; see
:func:`rotate_180`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# Metrica states both sample matches are played on a 105 x 68 m pitch.
PITCH_LENGTH_M: Final = 105.0
PITCH_WIDTH_M: Final = 68.0

# Laws of the Game: penalty area is 16.5 m deep and 40.32 m wide.
PENALTY_AREA_DEPTH_M: Final = 16.5
PENALTY_AREA_WIDTH_M: Final = 40.32
GOAL_WIDTH_M: Final = 7.32
SIX_YARD_DEPTH_M: Final = 5.5


@dataclass(frozen=True, slots=True)
class Pitch:
    """Pitch dimensions in metres, with the origin at the centre spot."""

    length: float = PITCH_LENGTH_M
    width: float = PITCH_WIDTH_M
    penalty_area_depth: float = PENALTY_AREA_DEPTH_M
    penalty_area_width: float = PENALTY_AREA_WIDTH_M

    @property
    def half_length(self) -> float:
        """Distance from the centre spot to either goal line."""
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        """Distance from the centre spot to either touchline."""
        return self.width / 2.0

    @property
    def attacking_goal_xy(self) -> tuple[float, float]:
        """Centre of the goal being attacked, in canonical coordinates."""
        return (self.half_length, 0.0)

    @property
    def penalty_area_x_min(self) -> float:
        """Canonical ``x`` at which the attacking penalty area begins."""
        return self.half_length - self.penalty_area_depth

    @property
    def penalty_area_y_abs_max(self) -> float:
        """Maximum ``|y|`` still inside the penalty area."""
        return self.penalty_area_width / 2.0

    def to_canonical(self, xy_unit: np.ndarray) -> np.ndarray:
        """Convert source ``[0, 1]`` coordinates to canonical metres.

        Args:
            xy_unit: Array with the final axis of size 2 holding ``(x, y)`` in
                Metrica's unit square. ``NaN`` entries propagate unchanged.

        Returns:
            Array of the same shape in canonical metres.
        """
        if xy_unit.shape[-1] != 2:
            msg = f"expected trailing axis of size 2, got shape {xy_unit.shape}"
            raise ValueError(msg)
        out = np.empty_like(xy_unit, dtype=np.float64)
        out[..., 0] = (xy_unit[..., 0] - 0.5) * self.length
        # Source y grows downward; canonical y grows upward.
        out[..., 1] = (0.5 - xy_unit[..., 1]) * self.width
        return out

    def is_inside_penalty_area(self, xy: np.ndarray) -> np.ndarray:
        """Test membership of the *attacking* penalty area (the ``+x`` end).

        Callers must pass coordinates already oriented so the attacking
        direction is ``+x``. ``NaN`` positions return ``False``.

        Args:
            xy: Canonical coordinates with a trailing axis of size 2.

        Returns:
            Boolean array with the trailing axis removed.
        """
        x = xy[..., 0]
        y = xy[..., 1]
        with np.errstate(invalid="ignore"):
            inside = (
                (x >= self.penalty_area_x_min)
                & (x <= self.half_length)
                & (np.abs(y) <= self.penalty_area_y_abs_max)
            )
        return np.asarray(inside & np.isfinite(x) & np.isfinite(y))

    def is_on_pitch(self, xy: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        """Test whether canonical coordinates lie within the pitch rectangle.

        Args:
            xy: Canonical coordinates with a trailing axis of size 2.
            tolerance: Extra margin in metres. Tracking providers routinely
                report players a little beyond the touchline, so callers
                validating plausibility should allow a metre or two.

        Returns:
            Boolean array with the trailing axis removed. ``NaN`` gives ``False``.
        """
        x = xy[..., 0]
        y = xy[..., 1]
        with np.errstate(invalid="ignore"):
            on = (np.abs(x) <= self.half_length + tolerance) & (
                np.abs(y) <= self.half_width + tolerance
            )
        return np.asarray(on & np.isfinite(x) & np.isfinite(y))

    def distance_to_attacking_goal(self, xy: np.ndarray) -> np.ndarray:
        """Euclidean distance in metres to the centre of the attacked goal."""
        gx, gy = self.attacking_goal_xy
        return np.asarray(np.hypot(xy[..., 0] - gx, xy[..., 1] - gy))

    def goal_visible_angle(self, xy: np.ndarray) -> np.ndarray:
        """Angle in radians subtended by the attacked goalmouth.

        A standard shot-quality primitive: the wider the goal appears from a
        position, the more inviting the situation. Returns ``0`` for positions
        level with or behind the goal line.

        Args:
            xy: Canonical coordinates with a trailing axis of size 2.

        Returns:
            Angle in radians, with the trailing axis removed.
        """
        gx = self.half_length
        dx = gx - xy[..., 0]
        y = xy[..., 1]
        half_goal = GOAL_WIDTH_M / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.arctan2(half_goal - y, dx)
            b = np.arctan2(-half_goal - y, dx)
            angle = np.abs(a - b)
            angle = np.where(dx > 0.0, angle, 0.0)
        return np.asarray(np.where(np.isfinite(angle), angle, 0.0))


def rotate_180(xy: np.ndarray) -> np.ndarray:
    """Rotate canonical coordinates 180 degrees about the centre spot.

    This is the only correct way to flip playing direction. Negating ``x``
    alone would mirror the pitch, swapping the left and right wings and
    inverting the handedness of any angular feature.

    Args:
        xy: Canonical coordinates with a trailing axis of size 2.

    Returns:
        Rotated coordinates of the same shape; ``NaN`` propagates.
    """
    return -np.asarray(xy, dtype=np.float64)


DEFAULT_PITCH: Final = Pitch()
