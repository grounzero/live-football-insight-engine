# ADR 0005: Set the operating point from a false-alarm budget

**Status:** accepted · **Date:** 2026-08-01 · **Supersedes:** window-precision targeting

## Context

The first threshold rule picked the lowest threshold reaching 0.30 window precision. Measured on
the training matches, that produced roughly **140 false alarms per 90 minutes**, about one every
forty seconds. Every model looked acceptable per window and unusable as a product.

Window precision optimises the wrong thing. Windows are not what a viewer experiences.

## Decision

Choose the lowest threshold whose measured false-alarm rate on the training matches stays within a
budget (default 12 per 90 minutes), and take whatever recall that affords. Still chosen on training
matches only, then frozen.

## Consequences

- The comparison changed completely. Under the budget the GRU reaches 0.76 episode precision at
  4–14 false alarms per 90, while the baselines cannot get below 25–80 without collapsing: their
  probabilities flicker across the threshold and shatter into many short alarms.
- Recall fell to 0.34–0.39 and median warning time to ~0.6 s. Both are real costs, reported in the
  README and model card rather than traded away quietly.
- The budget is configuration, so the precision/recall trade can be moved deliberately rather than
  discovered by accident.
