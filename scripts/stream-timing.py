#!/usr/bin/env python3
"""Measure what a browser actually receives from the insight stream.

Observational, and deliberately not a test. The numbers that matter here —
arrival jitter, burst density, time to the next insight — are properties of a
network and a host, so asserting on them in CI would make the suite fail
whenever someone else's infrastructure had a bad minute. What this is for is
comparing the same build across three places:

    python scripts/stream-timing.py --url http://127.0.0.1:8000
    python scripts/stream-timing.py --url http://127.0.0.1:8087      # container
    python scripts/stream-timing.py --url https://<app>.up.railway.app

The comparison is the point. A cadence that is identical locally and remote
says the producer is fine and the transport is not, which is a different fix
from a producer that cannot keep its own schedule — and the two are
indistinguishable from watching the pitch.

Standard library only, so it runs against a deployment from any checkout
without installing the package.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

#: Gaps below this are a single burst rather than two deliveries: two frames
#: that arrive in the same read cannot have been drawn separately.
BURST_MS = 2.0

#: A gap wide enough that a client without a playout buffer visibly stalls.
STALL_MS = 150.0


@dataclass
class Sample:
    """One message, as the browser would have seen it."""

    arrived_s: float
    kind: str
    bytes: int
    payload: dict[str, Any]


@dataclass
class Silence:
    """A stretch with no qualified insight, in wall seconds."""

    start_s: float
    end_s: float

    @property
    def length_s(self) -> float:
        """How long the silence lasted."""
        return self.end_s - self.start_s


@dataclass
class Report:
    """Everything one capture measured."""

    samples: list[Sample] = field(default_factory=list)

    def of(self, kind: str) -> list[Sample]:
        """Every captured message of one type, in arrival order."""
        return [s for s in self.samples if s.kind == kind]


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not values:
        return 0.0
    index = min(len(values) - 1, int(fraction * len(values)))
    return values[index]


def capture(url: str, seconds: float, timeout: float) -> Report:
    """Read the SSE stream for a bounded time, timestamping every message.

    Args:
        url: Base URL of the service.
        seconds: How long to listen.
        timeout: Socket timeout, so an unreachable host fails rather than hangs.

    Returns:
        The captured messages.

    Raises:
        SystemExit: If the stream cannot be opened.
    """
    endpoint = url.rstrip("/") + "/insights/stream"
    request = urllib.request.Request(endpoint, headers={"Accept": "text/event-stream"})
    report = Report()
    started = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                now = time.perf_counter()
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("data:"):
                    body = line[5:].strip()
                    try:
                        message = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    report.samples.append(
                        Sample(
                            arrived_s=now - started,
                            kind=str(message.get("type", "?")),
                            bytes=len(body),
                            payload=message.get("payload") or {},
                        )
                    )
                if now - started > seconds:
                    break
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        sys.exit(f"could not read {endpoint}: {exc}")

    if not report.samples:
        sys.exit(f"{endpoint} produced no messages in {seconds:.0f}s")
    return report


def _interarrivals(frames: list[Sample]) -> list[float]:
    """Gaps between consecutive visual frames, in milliseconds, sorted."""
    return sorted(
        (frames[i].arrived_s - frames[i - 1].arrived_s) * 1000.0 for i in range(1, len(frames))
    )


def _silences(report: Report, window_s: float) -> list[Silence]:
    """Stretches with no insight, including the head and tail of the capture.

    The head is included on purpose: it is what a visitor arriving at that
    moment would have waited, which is the number the product target is about.
    A measure that only counted gaps *between* insights would report a stream
    that opened with a 40 s wait as having no silence at all.
    """
    marks = [0.0, *[s.arrived_s for s in report.of("insight")], window_s]
    return [Silence(marks[i - 1], marks[i]) for i in range(1, len(marks))]


def _frame_report(report: Report, window_s: float) -> list[str]:
    """Cadence and jitter, the transport half of the picture."""
    frames = report.of("frame")
    lines = [f"  frames                 {len(frames)}"]
    if len(frames) < 2:
        return [*lines, "  (too few frames to measure a cadence)"]

    gaps = _interarrivals(frames)
    span = frames[-1].arrived_s - frames[0].arrived_s
    lines += [
        f"  visual rate            {len(frames) / span:.2f} Hz",
        f"  interarrival ms        p50 {percentile(gaps, 0.50):6.1f}"
        f"   p90 {percentile(gaps, 0.90):6.1f}"
        f"   p95 {percentile(gaps, 0.95):6.1f}"
        f"   p99 {percentile(gaps, 0.99):6.1f}",
        f"  interarrival ms        min {gaps[0]:6.1f}   max {gaps[-1]:6.1f}",
        f"  bursts (<{BURST_MS:.0f} ms apart) {sum(1 for g in gaps if g < BURST_MS)}"
        f"  of {len(gaps)}",
        f"  stalls (>{STALL_MS:.0f} ms apart) {sum(1 for g in gaps if g > STALL_MS)}"
        f"  of {len(gaps)}",
        f"  mean payload           {sum(f.bytes for f in frames) // len(frames)} bytes",
        f"  bandwidth              {sum(f.bytes for f in frames) / span / 1024:.1f} kB/s",
    ]

    # Ordering is only an invariant *within* one fixture and lap. A rotation or
    # a wrap restarts the source clock at zero and reuses the same frame
    # numbers, so checking the whole capture reports a fault every time the
    # demo does the thing it is supposed to do.
    segments: list[list[Sample]] = []
    previous_key: tuple[object, object] | None = None
    for frame in frames:
        key = (frame.payload.get("fixture"), frame.payload.get("lap"))
        if key != previous_key:
            segments.append([])
            previous_key = key
        segments[-1].append(frame)

    disordered = []
    for segment in segments:
        times = [float(f.payload.get("match_time_s", 0.0)) for f in segment]
        ids = [int(f.payload.get("frame", 0)) for f in segment]
        if times != sorted(times) or ids != sorted(ids) or len(set(ids)) != len(ids):
            disordered.append(segment[0].payload.get("fixture", "?"))

    # Measured per segment and averaged, for the same reason.
    rates = [
        (
            float(s[-1].payload.get("match_time_s", 0.0))
            - float(s[0].payload.get("match_time_s", 0.0))
        )
        / (s[-1].arrived_s - s[0].arrived_s)
        for s in segments
        if len(s) > 1 and s[-1].arrived_s > s[0].arrived_s
    ]
    fixtures = [s[0].payload.get("fixture", "?") for s in segments]
    order = "monotonic" if not disordered else f"OUT OF ORDER in {disordered}"
    speed = f"{sum(rates) / len(rates):.2f}x" if rates else "n/a"
    reported = frames[-1].payload.get("speed", "?")
    lines += [
        f"  segments               {len(segments)} (fixture/lap changes: {len(segments) - 1})",
        f"  ordering within each   {order}",
        f"  effective replay speed {speed}  (reported {reported}x)",
        f"  fixtures seen          {' -> '.join(dict.fromkeys(str(f) for f in fixtures))}",
    ]
    _ = window_s
    return lines


def _insight_report(report: Report, window_s: float) -> list[str]:
    """Time to value, the product half of the picture."""
    insights = report.of("insight")
    silences = _silences(report, window_s)
    lengths = sorted(s.length_s for s in silences)
    lines = [
        f"  insights               {len(insights)}",
        f"  suppression rollups    {len(report.of('suppression'))}",
    ]
    if insights:
        lines.append(f"  first insight after    {insights[0].arrived_s:.1f}s")
    else:
        lines.append(f"  first insight after    none in {window_s:.0f}s")
    if lengths:
        lines += [
            f"  silent interval s      p50 {percentile(lengths, 0.50):5.1f}"
            f"   p90 {percentile(lengths, 0.90):5.1f}"
            f"   max {lengths[-1]:5.1f}",
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    """Capture a stream and print the report.

    Returns:
        Process exit status; always 0 on a successful capture, because this
        measures rather than judges.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the service.")
    parser.add_argument("--seconds", type=float, default=30.0, help="How long to listen.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Socket timeout.")
    parser.add_argument("--json", action="store_true", help="Emit the raw samples as JSON.")
    args = parser.parse_args(argv)

    report = capture(args.url, args.seconds, args.timeout)
    window = report.samples[-1].arrived_s

    if args.json:
        json.dump(
            [
                {"t": s.arrived_s, "type": s.kind, "bytes": s.bytes, "payload": s.payload}
                for s in report.samples
            ],
            sys.stdout,
        )
        return 0

    kinds: dict[str, int] = {}
    for sample in report.samples:
        kinds[sample.kind] = kinds.get(sample.kind, 0) + 1

    print(f"\n{args.url}  —  {window:.1f}s, {len(report.samples)} messages")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print("\ntransport")
    print("\n".join(_frame_report(report, window)))
    print("\ntime to value")
    print("\n".join(_insight_report(report, window)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
