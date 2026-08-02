# ADR 0003: Infer playing direction from ranked evidence, and fail loudly

**Status:** accepted · **Date:** 2026-08-01

## Context

Getting playing direction wrong inverts every spatial feature, and does so silently: the pipeline
runs, the model trains, the results are meaningless.

The three sample matches are not uniform. Game 3 (EPTS) declares `attack_direction_first_half` and
labels goalkeepers. Games 1 and 2 (CSV) declare neither, and the two disagree with each other,
the home team attacks `+x` in the first half of game 1 and `−x` in the first half of game 2.

## Decision

Four tiers of evidence vote with a weight of `tier weight × margin`:

| Tier | Signal | Weight |
|---|---|---|
| 1 | provider metadata | 1.00 |
| 2 | mean pass/carry progression | 0.75 |
| 3 | goalkeeper position; team centroid depth | 0.65 |
| 4 | shot geometry | 0.35 |

Below 70% weighted agreement, or when a high-volume signal contradicts declared metadata,
preparation **fails** with every signal listed. Two structural facts are enforced regardless of the
vote: the two teams cannot attack the same end, and a team must change ends at half time.

An override exists but requires a written reason, which is echoed into the audit artifact.

## Consequences

- Shot geometry is corroboration, not evidence. One team-period in game 3 has three shots; pass
  progression draws on hundreds of events and carries that period.
- Game 3 becomes a ground-truth fixture. With its declaration withheld, the inference reproduces it
  for all four team-periods, which is direct evidence that the inference used on games 1 and 2 is
  right.
- The audit artifact is written before any failure, so a rejected match is still diagnosable.
- Development cost: the structural check caught a genuine sign error in the team-centroid signal
  (a team is anchored near the goal it *defends*, so it attacks away from its centroid). A single
  unchecked signal would have shipped that.
