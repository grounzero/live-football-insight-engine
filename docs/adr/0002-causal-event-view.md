# ADR 0002: Enforce causality with a type, not a convention

**Status:** accepted · **Date:** 2026-08-01

## Context

Temporal leakage is the failure mode most likely to make every reported number wrong while every
test still passes. The obvious form, using a future event, is easy to avoid. The subtle form is
not: a pass event carries the identity of the player who eventually receives it and where the ball
lands. Reading those fields mid-flight tells the model the pass completes.

A code-review convention ("don't look at the future") does not survive contact with a codebase that
grows.

## Decision

Feature builders never receive the event list. They receive a `CausalEventView`, which answers
questions only about a specific instant and enforces two rules:

1. **Visibility.** An event does not exist until `start_frame <= now`.
2. **Resolution.** A visible event whose `end_frame` is still in the future is *in flight*:
   `end_frame`, `end_time_s`, `end_xy` and `to_player` are `None` until it resolves.

## Consequences

- Leakage becomes a `None` at runtime and a type error under mypy, rather than something a reviewer
  must notice.
- Possession lags real turnovers, because it can only change once the next event has started. That
  is what a live system would experience, so it is correct rather than a limitation to fix.
- Features are measurably weaker than a hindsight version would produce. That was expected and
  accepted; the alternative is optimistic numbers.
- Four tests assert the property directly, including deleting, mutating and fabricating future
  events and tampering with an in-flight outcome.
