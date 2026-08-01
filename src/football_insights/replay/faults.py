"""Deterministic fault injection for replay.

Testing a live system against a clean feed proves very little. These profiles
reproduce what real tracking feeds actually do — jitter, gaps, duplicates, late
frames, frames arriving out of order — so the ingestion and suppression paths
are exercised against the conditions they exist for.

**The determinism contract.** The emitted stream is a pure function of
``(profile, seed, source frames)``. Two rules make that true:

1. One :class:`random.Random` instance, owned here. Never the global module RNG
   and never numpy's, either of which could be perturbed by unrelated code.
2. A fixed number of draws per source frame, consumed in a fixed order,
   *whether or not the outcome fires*. Drawing conditionally would make the
   sequence depend on earlier outcomes, so a single changed probability would
   shift every later frame.

Wall-clock time influences only *when* a frame is released, never *which* frames
are released or in what order. That is what lets a test assert byte-identical
streams while replaying at 50x speed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from football_insights.types import JsonDict

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from football_insights.config import FaultProfileSettings
    from football_insights.domain import Frame

#: Draws consumed per source frame, in this order. Kept as a named constant so
#: the determinism test can assert the count never changes silently.
DRAWS_PER_FRAME = 6


@dataclass(frozen=True, slots=True)
class EmittedFrame:
    """A frame as the replay layer released it.

    Attributes:
        offset_s: Extra delay applied on top of the frame's natural schedule,
            combining jitter and any injected delay.
        is_duplicate: Whether this is a repeat of the preceding frame.
        was_reordered: Whether this frame was deliberately released late,
            arriving after frames that followed it in the source.
        sequence: Position in the emitted stream, for assertions and logging.
    """

    frame: Frame
    offset_s: float
    is_duplicate: bool = False
    was_reordered: bool = False
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class FaultSummary:
    """What a fault profile actually did to a stream."""

    profile: str
    seed: int
    source_frames: int
    emitted_frames: int
    dropped: int
    duplicated: int
    delayed: int
    reordered: int

    def to_dict(self) -> JsonDict:
        """Serialisable form for `/replay/status` and test reports."""
        return {
            "profile": self.profile,
            "seed": self.seed,
            "source_frames": self.source_frames,
            "emitted_frames": self.emitted_frames,
            "dropped": self.dropped,
            "duplicated": self.duplicated,
            "delayed": self.delayed,
            "reordered": self.reordered,
        }


class FaultInjector:
    """Applies a fault profile to a frame sequence, reproducibly.

    Args:
        profile: The profile to apply.
        seed: Seed for this run. Identical seeds give identical streams.
    """

    __slots__ = ("_profile", "_seed")

    def __init__(self, profile: FaultProfileSettings, seed: int) -> None:
        """Bind a profile to a seed."""
        self._profile = profile
        self._seed = seed

    @property
    def profile_name(self) -> str:
        """Name of the active profile."""
        return self._profile.name

    @property
    def seed(self) -> int:
        """Seed for this run."""
        return self._seed

    def apply(self, frames: Iterable[Frame]) -> tuple[list[EmittedFrame], FaultSummary]:
        """Transform a frame sequence according to the profile.

        Args:
            frames: Source frames in order.

        Returns:
            The emitted stream and a summary of what was injected.
        """
        rng = random.Random(self._seed)
        p = self._profile
        source: Sequence[Frame] = list(frames)

        dropped = duplicated = delayed = reordered = 0
        staged: list[tuple[float, int, EmittedFrame]] = []

        for index, frame in enumerate(source):
            # Fixed draw order, always consumed. See the module docstring.
            r_drop = rng.random()
            r_duplicate = rng.random()
            r_delay = rng.random()
            delay_ms = rng.uniform(*p.delay_ms) if p.delay_ms[1] > 0 else 0.0
            r_reorder = rng.random()
            jitter_ms = rng.uniform(*p.jitter_ms) if p.jitter_ms[1] > 0 else 0.0

            if r_drop < p.drop_prob:
                dropped += 1
                continue

            offset = jitter_ms / 1000.0
            if r_delay < p.delay_prob:
                offset += delay_ms / 1000.0
                delayed += 1

            # Reordering shifts a frame's place in the output sequence rather
            # than its timestamp, which is how a real out-of-order arrival looks.
            order = float(index)
            was_reordered = False
            if r_reorder < p.reorder_prob and p.reorder_window > 0:
                order += p.reorder_window
                was_reordered = True
                reordered += 1

            staged.append((order, index, EmittedFrame(frame, offset, False, was_reordered)))
            if r_duplicate < p.duplicate_prob:
                duplicated += 1
                staged.append((order, index, EmittedFrame(frame, offset, True, was_reordered)))

        # Stable sort on (order, source index) keeps the transformation total
        # and reproducible; equal keys retain insertion order.
        staged.sort(key=lambda item: (item[0], item[1]))
        emitted = [
            EmittedFrame(
                frame=item[2].frame,
                offset_s=item[2].offset_s,
                is_duplicate=item[2].is_duplicate,
                was_reordered=item[2].was_reordered,
                sequence=position,
            )
            for position, item in enumerate(staged)
        ]

        summary = FaultSummary(
            profile=p.name,
            seed=self._seed,
            source_frames=len(source),
            emitted_frames=len(emitted),
            dropped=dropped,
            duplicated=duplicated,
            delayed=delayed,
            reordered=reordered,
        )
        return emitted, summary


def stream_signature(emitted: Sequence[EmittedFrame]) -> tuple[tuple[int, int, bool, bool], ...]:
    """Compact, comparable fingerprint of an emitted stream.

    Used by the determinism tests: comparing signatures gives a readable failure
    when two runs diverge, rather than a wall of frame objects.

    Args:
        emitted: The emitted stream.

    Returns:
        One tuple per emitted frame of
        ``(frame id, offset in microseconds, is duplicate, was reordered)``.
    """
    return tuple(
        (e.frame.frame, round(e.offset_s * 1_000_000), e.is_duplicate, e.was_reordered)
        for e in emitted
    )
