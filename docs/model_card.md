# Model card — penalty-area entry predictor

Last updated with the reference run of 2026-08-01. Reproduce with `make reference`.

## Intended use

Generating short-horizon, clearly qualified attacking-threat insights for a football viewing
experience, under human editorial oversight, on tracking data of the same kind and quality as the
one it was trained on.

It is intended as a **suggestion engine with a suppression layer**, not an autonomous commentator.
The editorial stage exists because a statistically valid prediction is not automatically something
worth showing, and it is expected to remain in place.

## Not intended for

- Any decision affecting a person: recruitment, valuation, selection, medical or disciplinary.
- Predicting injury, intent, effort, or any personal or physiological characteristic. The model has
  no access to such information and makes no such claim.
- Betting markets or any use where a miscalibrated probability carries financial consequence. The
  probabilities are **not** well calibrated (see below).
- Officiating, or any use where output is treated as a factual record.
- Deployment on tracking from a different provider, league or season without re-validation. Three
  matches from one provider cannot support a generalisation claim.
- Unattended operation with output published directly to an audience.

## Task and label

At prediction time `t`, positive if and only if the ball — with the attacking team in possession —
crosses from outside into the opposition penalty area during `(t, t + 5 s]`.

- Observation window 5 s, horizon 5 s, stride 0.5 s, resampled to 10 Hz (50 timesteps). All
  configurable.
- Penalty area per the Laws of the Game: 16.5 m deep, 40.32 m wide, on a 105 × 68 m pitch.
- Entries are detected from **tracking** at 25 Hz, so the crossing instant is precise; possession
  comes from **event annotation**, causally.
- Entries by the same team within 10 s merge into one *episode*.

Excluded from sampling, each for a reason: too early in a period for a full window; ball out of
play; possession unknown; ball already inside the area (the event has already happened).

**Why this target.** Chosen from the data, not preference. The dataset holds 24 shots per match
against 50–60 box entries; a held-out match with 24 positives cannot support a usable precision
estimate. The trade is real: a box entry is not danger. It ignores long-range shots entirely and
scores a tame ball into the corner of the area identically to a cut-back to the penalty spot.

## Training data

Metrica Sports sample data — three anonymised matches, 25 Hz optical tracking with synchronised
events. 27,557 samples, 1,598 positive (5.80%), 196 episodes. See
[data_card.md](data_card.md) for provenance, licensing and known quality issues.

Splits are **match-aware**. With a 0.5 s stride, adjacent windows overlap almost entirely; a random
split would place near-duplicates of one attack on both sides. Early stopping uses a time-ordered
tail of the training matches with a 30 s embargo either side of the cut.

## Metrics

Leave-one-match-out, thresholds fixed on training matches to a budget of ≤12 false alarms / 90 min:

| Model | Window PR-AUC (folds) | Episode precision | Episode recall | FA / 90 |
|---|---|---|---|---|
| gru-temporal | 0.530 / 0.569 / 0.457 | 0.678 | 0.426 | 18 / 14 / 6 |
| baseline-logistic | 0.545 / 0.506 / 0.445 | 0.574 | 0.655 | 38 / 23 / 30 |
| baseline-gbdt | 0.541 / 0.500 / 0.415 | 0.483 | 0.804 | 51 / 76 / 38 |
| heuristic-fallback *(not ML)* | 0.467 / 0.392 / 0.307 | 0.407 | 0.121 | 12 / 6 / 14 |

Reference run (train games 1 + 3, test game 2), GRU: PR-AUC 0.569, Brier 0.128, episode precision
0.706 [0.61, 0.81], episode recall 0.590, 14.5 false alarms / 90 min, median warning 1.22 s,
threshold 0.93.

**On the baseline comparison, stated plainly:** on window PR-AUC the GRU's advantage over the
baselines is marginal — the folds overlap, logistic regression wins one of three, and the mean gap
(0.519 vs 0.499) sits inside the fold-to-fold spread. The GRU's real advantage is confined to the
operating point the product needs, where it holds 0.68 episode precision at 6–18 false alarms per
90 minutes against 23–76 for the baselines, because its probabilities are smoother in time and
produce fewer, more coherent alarms.

Episode-level metrics are the headline because window metrics count roughly ten near-identical
views of the same attack as ten observations. Both are reported.

## Threshold selection

Chosen on training matches only, then frozen before the held-out match is scored.

The criterion is a **false-alarm budget**, not a window-precision target. Targeting window
precision of 0.30 was measured to produce ~140 false alarms per 90 minutes — roughly one every
forty seconds, which no viewer-facing product could use. The binding constraint is how often the
system may interrupt, so that is what the threshold is set from; recall is whatever the budget
affords.

The episode-grouping knobs (`merge_gap_s` 10 s, `alarm_bridge_gap_s` 2 s) materially move episode
precision and are frozen the same way, recorded in the run config and hashed into the config
fingerprint.

## Calibration

**Poor, and the worst of the four models.** Brier 0.128 for the GRU against 0.051 for gradient
boosting. It is trained with a positive-class weight of roughly 16:1 to counter the imbalance,
which deliberately inflates output probabilities. Rankings are meaningful; the numbers are not
frequencies. A reliability curve is written to `artifacts/reports/reference_run.json`.

Do not surface the raw probability to an audience as a percentage. The demo shows it as a
confidence bar with the decision threshold marked, which is a comparison against an operating
point, not a claim about likelihood.

## Known limitations and failure modes

- **Warning time 1.22 s median** — enough to precede the event, still short for a broadcast
  insight. This is the direct cost of buying precision with a high threshold and is the model's
  main practical weakness.
- **Recall 0.59** — four in ten box entries pass unflagged.
- **The threshold does not transfer exactly.** Fitted to ≤12 false alarms / 90 min on two matches,
  it produced 14.5 on the third. A useful reminder of how little two matches constrain an operating
  point.
- **Three matches, 196 episodes.** Fold spread is wide; no interval captures between-match variance.
- **Bootstrap recall intervals are biased low** — resampled duplicate episodes cannot all be
  matched under one-to-one alarm matching. Precision and false-alarm intervals are unaffected.
- **Set pieces and restarts** are excluded from training, so behaviour around them is untested.
- **Possession lags** real turnovers, because it can only change once the next event has started.
- **Substitutions and missing players** leave `NaN` columns; features degrade gracefully but the
  goalkeeper-dependent ones lose meaning if the keeper is untracked.
- Sustained possession in the final third will keep the model near threshold; the cooldown, not the
  model, is what prevents repeated insights.

## Risks of presenting predictions as fact

A short-horizon probability rendered as a statement ("they're about to get in behind") will be
wrong often enough to damage trust, and a viewer has no way to tell an unlucky miss from a bad
model. Three mitigations are built in rather than left to policy:

1. Every template is hedged, and a test enforces it against a lexicon of hedge terms and forbidden
   assertions.
2. Factual context ("3 attackers ahead of the ball") is measured from the same window and kept
   separate from the hedged headline, so the insight is concrete without over-claiming.
3. The system emits nothing when it cannot support a claim — invalid window, missing model, schema
   mismatch, ball out of play — and records why.

## Human oversight

Intended to run with an editor able to see the suppression counters and mute the feed. The
`fi_insight_suppressed_total{reason}` series is the operational signal: a shift in its composition
usually means the input data changed, not that the model improved.

## Why the system suppresses output

Withholding is the common case, not an error path. In a full match under a degraded feed the engine
scores tens of thousands of windows and emits around twenty insights. Reasons: `low_confidence`,
`invalid_window`, `insufficient_frames`, `cooldown`, `duplicate_recent`, `stale_situation`,
`already_in_box`, `dead_ball`, `not_yet_sustained`, `model_unavailable`, `schema_mismatch`.

Each is counted separately so that "the model is quiet" and "the system is broken" are never
confused.

## Provenance

Every artifact records model version, training timestamp, git revision, feature-schema hash,
dataset fingerprint, config fingerprint, training matches, decision threshold and test metrics
(`artifacts/registry/*.metadata.json`). The service refuses to load a model whose feature-schema
hash disagrees with the running code.
