# ADR 0001 — Predict penalty-area entries, not shots

**Status:** accepted · **Date:** 2026-08-01

## Context

The brief allows several targets: a shot, a touch in the box, a penalty-area entry, or an action
entering a high-danger zone. The choice had to be made from what the data can actually support.

Counting the source events before committing: **24 shots per match** across both CSV matches
(48 total, 72 including game 3), against **50–63 penalty-area entries per match**.

## Decision

Predict whether the ball, with the attacking team in possession, enters the opposition penalty area
within the next 5 seconds.

## Consequences

- 196 independent episodes across three matches instead of ~72. A held-out match contains ~60
  positives rather than ~24, which is the difference between a precision estimate that means
  something and one dominated by sampling noise.
- Window-level positive rate is 5.80%, imbalanced but trainable.
- The cost is honest and permanent: **a box entry is not danger.** It excludes long-range shots
  entirely and treats a tame ball into the corner of the area the same as a cut-back. This is
  stated in the model card rather than glossed.
- Shots remain available in the event stream if a future version wants a two-stage target.
