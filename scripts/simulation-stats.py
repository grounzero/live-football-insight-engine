#!/usr/bin/env python3
"""Measure what the generated fixtures actually look like.

The counterpart to ``scripts/calibrate-from-metrica.py``. That script reads the
reference dataset and prints the distributions the simulation is calibrated
against; this one runs the simulation and prints the same statistics for it, so
the two can be put side by side.

It exists because "the movement looks artificial" is not a reviewable claim. The
goalkeeper problem this was written for looked like a rendering quirk and was in
fact a keeper whose forward target was a constant: nought metres of travel
across a whole period, which is obvious in a number and arguable in a video.

Runs the simulation only. No dataset, no network, no model.

Usage:
    python scripts/simulation-stats.py
    python scripts/simulation-stats.py --minutes 10 --profile counter
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

import numpy as np

from football_insights.data.profiles import PROFILES, PROFILES_BY_KEY
from football_insights.data.simulation import N_PLAYERS, simulate_period
from football_insights.domain import Team

#: Reference values, for reading the simulated column against something. Player
#: speed and possession duration are measured from the Metrica sample matches by
#: the calibration script; distance covered and event counts are the widely
#: published ranges for professional matches.
REFERENCE = {
    "gk_distance_km90": "4.0-5.0",
    "gk_still_pct": "~35",
    "outfield_speed_median": "1.50",
    "outfield_distance_km90": "10-11",
    "possession_median_s": "4.10",
    "deflections_90": "tens",
    "events_90": "1700-2000",
}

#: Below this a player is standing rather than moving.
STILL_MS = 0.2


def _period(profile_key: str, minutes: float, frame_rate: float, seed: int) -> object:
    """Simulate one period of a profile."""
    profile = PROFILES_BY_KEY[profile_key]
    return simulate_period(
        random.Random(seed),
        profile,
        period=1,
        n_frames=int(minutes * 60 * frame_rate),
        frame_offset=0,
        start_s=0.0,
        frame_rate=frame_rate,
        attack_sign={Team.HOME: 1.0, Team.AWAY: -1.0},
    )


def _motion(xy: np.ndarray, frame_rate: float, per_90: float) -> dict[str, float]:
    """Speed and distance statistics for one team's tracking."""
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=2) * frame_rate
    gk, outfield = speed[:, 0], speed[:, 1:]
    gk_distance = float(np.linalg.norm(np.diff(xy[:, 0], axis=0), axis=1).sum())
    out_distance = float(np.linalg.norm(np.diff(xy[:, 1:], axis=0), axis=2).sum() / (N_PLAYERS - 1))
    return {
        "gk_speed_median": float(np.median(gk)),
        "gk_still_pct": float(np.mean(gk < STILL_MS) * 100.0),
        "gk_distance_km90": gk_distance * per_90 / 1000.0,
        "gk_x_span": float(xy[:, 0, 0].max() - xy[:, 0, 0].min()),
        "gk_y_span": float(xy[:, 0, 1].max() - xy[:, 0, 1].min()),
        "out_speed_median": float(np.median(outfield)),
        "out_speed_p95": float(np.percentile(outfield, 95)),
        "out_distance_km90": out_distance * per_90 / 1000.0,
    }


def _spells(events: list[object]) -> list[float]:
    """Seconds each side held the ball before the other side did."""
    spells: list[float] = []
    current: object | None = None
    started: float | None = None
    for event in events:
        team = event.team  # type: ignore[attr-defined]
        if team != current:
            if current is not None and started is not None:
                spells.append(event.start_time_s - started)  # type: ignore[attr-defined]
            current, started = team, event.start_time_s  # type: ignore[attr-defined]
    return spells


def main(argv: list[str] | None = None) -> int:
    """Measure every profile and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=5.0, help="match time per profile")
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--profile", action="append", choices=[p.key for p in PROFILES])
    args = parser.parse_args(argv)

    keys = args.profile or [p.key for p in PROFILES]
    per_90 = 90.0 * 60.0 / (args.minutes * 60.0)

    print(f"{args.minutes:g} minutes per profile at {args.frame_rate:g} Hz, seed {args.seed}\n")
    for index, key in enumerate(keys):
        period = _period(key, args.minutes, args.frame_rate, args.seed + index)
        print(f"── {key}")
        for label, xy in (("home", period.home_xy), ("away", period.away_xy)):
            stats = _motion(xy, args.frame_rate, per_90)
            print(
                f"   {label}  GK median {stats['gk_speed_median']:4.2f} m/s   "
                f"still {stats['gk_still_pct']:5.1f}%   "
                f"{stats['gk_distance_km90']:4.2f} km/90   "
                f"span x {stats['gk_x_span']:5.1f} y {stats['gk_y_span']:5.1f} m"
            )
            print(
                f"         outfield median {stats['out_speed_median']:4.2f} m/s   "
                f"p95 {stats['out_speed_p95']:4.2f}   "
                f"{stats['out_distance_km90']:5.2f} km/90"
            )

        spells = _spells(period.events)
        kinds = Counter(
            f"{e.type.value}/{e.subtype or '-'}"
            for e in period.events  # type: ignore[attr-defined]
        )
        deflections = sum(v for k, v in kinds.items() if "DEFLECTION" in k)
        # Metrica's schema has no reception event — a completed pass carries its
        # receiver in `to_player`. Counting ours against that reference without
        # subtracting them would compare two different things.
        receptions = sum(v for k, v in kinds.items() if "RECEPTION" in k)
        comparable = len(period.events) - receptions
        print(
            f"   possession median {np.median(spells):5.2f} s   "
            f"p90 {np.percentile(spells, 90):5.2f} s   "
            f"(reference {REFERENCE['possession_median_s']})"
        )
        print(
            f"   events {comparable * per_90:6.0f}/90 comparable "
            f"({len(period.events) * per_90:.0f} incl. receptions)   "
            f"deflections {deflections * per_90:4.0f}/90   "
            f"box entries {len(period.true_entries) * per_90:4.0f}/90"
        )
        print("   " + ", ".join(f"{k}={v * per_90:.0f}" for k, v in kinds.most_common(6)))
        print()

    print("reference: " + "  ".join(f"{k}={v}" for k, v in REFERENCE.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
