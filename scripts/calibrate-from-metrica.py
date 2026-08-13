#!/usr/bin/env python3
"""Recompute the aggregate distributions the simulation is calibrated against.

The generated fixtures are not derived from the Metrica sample data — no
coordinate sequence, event timeline or identifier is copied into them. What is
taken from it is a handful of *distributions*: how fast players move, how hard
they accelerate, how far apart they stand, how long a side keeps the ball, how
long a pass is and how long it is in the air.

Those numbers are baked into
:mod:`football_insights.data.simulation` as documented constants. This script
recomputes them so the constants can be checked against their source rather than
taken on trust, and so anyone changing the simulation can see which reference
statistic a constant was fitted to.

It reads the locally downloaded dataset and prints percentiles. Nothing it
prints identifies a player, a match or a moment; that is the point.

Usage:
    python scripts/calibrate-from-metrica.py
    python scripts/calibrate-from-metrica.py --match Sample_Game_2
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

import numpy as np

from football_insights.config import Settings
from football_insights.domain import ON_BALL_TYPES, EventType
from football_insights.errors import DataValidationError
from football_insights.serving.loader import load_match

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Percentiles printed for every statistic. Deliberately coarse: these are
#: calibration ranges, and quoting them to four decimal places would imply the
#: simulation reproduces a distribution it only has to sit inside.
PERCENTILES = (5, 25, 50, 75, 90, 99)

DEFAULT_MATCHES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")


def _report(name: str, values: Sequence[float] | np.ndarray, unit: str) -> None:
    """Print one distribution as percentiles."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        print(f"  {name:26} (no samples)")
        return
    cells = "  ".join(f"p{p}={np.percentile(array, p):7.2f}" for p in PERCENTILES)
    print(f"  {name:26} {cells}   [{unit}, n={array.size}]")


def main(argv: list[str] | None = None) -> int:
    """Compute and print the calibration ranges.

    Returns:
        Process exit status; non-zero when the dataset is not available.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--match",
        action="append",
        dest="matches",
        help="Match id; repeatable. All three when omitted.",
    )
    args = parser.parse_args(argv)
    matches = tuple(args.matches) if args.matches else DEFAULT_MATCHES

    settings = Settings()
    speeds: list[np.ndarray] = []
    accels: list[np.ndarray] = []
    spacing: list[np.ndarray] = []
    ball_speeds: list[np.ndarray] = []
    possessions: list[float] = []
    pass_lengths: list[float] = []
    pass_flights: list[float] = []
    touch_gaps: list[float] = []

    for match_id in matches:
        try:
            tracking, events, _ = load_match(settings, match_id)
        except DataValidationError as exc:
            print(f"{exc}", file=sys.stderr)
            return 1

        rate = tracking.frame_rate
        dt = 1.0 / rate
        for team_xy in (tracking.home_xy, tracking.away_xy):
            # Outfield only: a goalkeeper's movement is a different
            # distribution and the simulation models them separately.
            outfield = team_xy[:, 1:, :]
            velocity = np.diff(outfield, axis=0) / dt
            speed = np.hypot(velocity[..., 0], velocity[..., 1])
            accel = np.hypot(*(np.diff(velocity, axis=0) / dt).transpose(2, 0, 1))
            speeds.append(speed[np.isfinite(speed)])
            accels.append(accel[np.isfinite(accel)])

            # Sampled every five seconds: nearest-neighbour spacing moves
            # slowly and the pairwise distance is quadratic in squad size.
            for index in range(0, outfield.shape[0], int(rate) * 5):
                present = outfield[index]
                present = present[np.isfinite(present).all(axis=1)]
                if len(present) < 4:
                    continue
                gaps = np.hypot(
                    present[:, None, 0] - present[None, :, 0],
                    present[:, None, 1] - present[None, :, 1],
                )
                np.fill_diagonal(gaps, np.inf)
                spacing.append(gaps.min(axis=1))

        ball_step = np.hypot(*np.diff(tracking.ball_xy, axis=0).T) * rate
        ball_speeds.append(ball_step[np.isfinite(ball_step)])

        on_ball = [e for e in events if e.type in ON_BALL_TYPES]
        run_team = None
        run_start = None
        for index, event in enumerate(on_ball):
            if event.team != run_team:
                if run_team is not None and run_start is not None:
                    possessions.append(on_ball[index - 1].end_time_s - run_start)
                run_team, run_start = event.team, event.start_time_s
            if index:
                touch_gaps.append(max(0.0, event.start_time_s - on_ball[index - 1].start_time_s))
            if event.type is EventType.PASS and event.start_xy and event.end_xy:
                pass_lengths.append(
                    float(
                        np.hypot(
                            event.end_xy[0] - event.start_xy[0],
                            event.end_xy[1] - event.start_xy[1],
                        )
                    )
                )
                pass_flights.append(max(0.0, event.end_time_s - event.start_time_s))

    print(f"\nAggregate calibration ranges from {', '.join(matches)}")
    print("Distributions only: nothing below identifies a player, match or moment.\n")
    _report("player speed", np.concatenate(speeds), "m/s")
    _report("player acceleration", np.concatenate(accels), "m/s^2")
    _report("nearest team-mate", np.concatenate(spacing), "m")
    _report("ball speed", np.concatenate(ball_speeds), "m/s")
    print()
    _report("possession duration", possessions, "s")
    _report("pass length", pass_lengths, "m")
    _report("pass flight time", pass_flights, "s")
    _report("gap between touches", touch_gaps, "s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
