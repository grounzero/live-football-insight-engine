# ADR 0004 — Score episodes, and bootstrap over possessions

**Status:** accepted · **Date:** 2026-08-01

## Context

A 0.5 s stride with a 5 s horizon means one penalty-area entry generates roughly ten positive
windows that overlap almost entirely. Treating them as ten independent observations inflates every
metric and, worse, every confidence interval.

## Decision

- Ground-truth entries by the same team within 10 s merge into one **episode**.
- Consecutive firing windows, bridging gaps up to 2 s, group into one **alarm**.
- An alarm detects an episode when it overlaps `[entry − horizon, entry]`; matching is greedy and
  one-to-one.
- Confidence intervals use a **cluster bootstrap over possession sequences**, never over windows.
- Alarms firing before the lead-up interval are reported separately as *early*, not silently
  counted as false alarms — a twelve-second warning is outside the contract but not football-wrong.

## Consequences

- Episode metrics became the headline, and they told a different story from window metrics: on
  PR-AUC the GRU does not beat the baselines, but at a fixed false-alarm budget it is the only
  model that can hold the budget at all.
- The grouping knobs move results, so they are frozen on training matches and hashed into the
  config fingerprint.
- **A known bias:** bootstrap *recall* intervals run low, because a resampled cluster can appear
  several times while matching stays one-to-one, so duplicated episodes cannot all be matched.
  Precision and false-alarm intervals are unaffected. Disclosed rather than quietly reported.
- With three matches, fold-to-fold spread remains the real measure of uncertainty; no interval can
  represent between-match variance.
