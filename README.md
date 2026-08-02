# Live Football Insight Engine

Replays recorded football tracking data as though it were arriving live, predicts whether the
attacking team will enter the opposition penalty area within the next few seconds, and turns that
prediction into a qualified, viewer-facing insight. Far more often, it decides to stay quiet.

**The viewer problem.** During a match, the interesting moment is usually a second or two *before*
anything happens: a full-back stepping up, three runners beyond the ball, space opening in behind.
A system that can flag "threat is building" a moment early is useful. A system that says it every
forty seconds is noise, and one that states an uncertain estimate as fact is worse than useless.
Most of the engineering here is about that second problem: turning a probability into something
worth showing, and knowing when not to.

> Not affiliated with any league, broadcaster or data provider. Predictions are estimates over a
> short horizon, never statements of fact.

---

## What it does

```
Metrica tracking (25 Hz)
   │
   ├─ parse ─ validate ─ infer playing direction (audited) ─ causal possession
   │
   ├─ label: does the ball enter the box in the next 5 s?
   │
   ├─ features: 39 identity-invariant spatial/temporal features per frame
   │
   ├─ model: GRU  ·  baselines: logistic + gradient boosting  ·  fallback: rule-based
   │
   ├─ EDITORIAL LAYER ─ threshold, cooldown, staleness, duplicate, data validity
   │
   └─ FastAPI  ─ SSE ─ React demo   +   Prometheus metrics
```

The two stages after the model are deliberately separate. `fi_model_*` metrics describe the
predictor; `fi_insight_*` metrics describe what a viewer actually experienced. A model behaving
normally while the editorial layer suppresses everything is a completely different incident from a
model that has stopped firing, and the metrics say which.

## Quick start

```bash
make setup      # venv + install (Python 3.11+; developed on 3.14)
make slice0     # end-to-end acceptance test on synthetic data, no download needed
make demo       # build the React demo and serve it at http://127.0.0.1:8000
```

`make slice0` and `make test` need no data at all: every test runs against a seeded synthetic
match generated in-process. To work with the real dataset:

```bash
make data       # download Metrica sample data (~180 MB, never committed)
make prepare    # parse, validate, orient, feature-ise, label
make train      # train baselines + GRU, register artifacts
make evaluate   # leave-one-match-out CV with cluster-bootstrap intervals
make benchmark  # PyTorch vs ONNX Runtime
```

Those five stages can also be started from the demo, one at a time and with streamed output, if
the service is run with `--dev-tools`. That is off by default and deliberately so; see
[The demo](#the-demo).

## The prediction task

> At time `t`, will the ball (with the attacking team in possession) cross into the opposition
> penalty area at some point in `(t, t + 5s]`?

**Why box entries and not shots.** The dataset has 24 shots per match but 50–60 penalty-area
entries. A held-out match containing 24 positives cannot support a precision estimate anyone
should act on. Box entries give 196 independent episodes across three matches: still small, but
enough to evaluate honestly. The cost is that a box entry is *not* danger: it excludes long-range
shots and counts a tame ball into the corner of the area the same as a cut-back. See
[docs/model_card.md](docs/model_card.md).

Observation window, horizon, stride and every threshold are configuration, not constants
([`config.py`](src/football_insights/config.py)).

## Results

Measured, not targeted. Reproduce with `make prepare && make evaluate`.

**Dataset**: 3 matches, 27,557 samples, 1,598 positives (5.80%), 196 episodes.

| Match | Samples | Positives | Rate | Episodes |
|---|---|---|---|---|
| Sample_Game_1 | 7,997 | 659 | 8.24% | 77 |
| Sample_Game_2 | 8,744 | 471 | 5.39% | 61 |
| Sample_Game_3 | 10,816 | 468 | 4.33% | 58 |

**Leave-one-match-out**, thresholds chosen on training matches to a budget of ≤12 false alarms per
90 minutes, then frozen:

| Model | Window PR-AUC (per fold) | Episode precision | Episode recall | False alarms / 90 |
|---|---|---|---|---|
| **gru-temporal** | 0.530 / 0.569 / 0.457 | **0.678** | 0.426 | **18 / 14 / 6** |
| baseline-logistic | 0.545 / 0.506 / 0.445 | 0.574 | 0.655 | 38 / 23 / 30 |
| baseline-gbdt | 0.541 / 0.500 / 0.415 | 0.483 | 0.804 | 51 / 76 / 38 |
| heuristic-fallback *(not ML)* | 0.467 / 0.392 / 0.307 | 0.407 | 0.121 | 12 / 6 / 14 |

**Does the neural model beat the baselines? On PR-AUC, barely.** The folds overlap: logistic
regression wins one fold outright, the GRU wins two, and the mean gap (0.519 vs 0.499) is well
inside the fold-to-fold spread. Anyone reading only the PR-AUC column should conclude the sequence
model is not clearly earning its complexity.

The separation appears at the operating point the product actually needs. Held to a false-alarm
budget, the GRU reaches 0.68 episode precision at 6–18 false alarms per 90 minutes, while the
baselines sit at 23–76 for lower precision. Their probabilities flicker across the threshold and
shatter into many short alarms; the GRU's outputs are smoother in time, so it produces fewer, more
coherent ones, which is what episode-level scoring rewards and window-level scoring cannot see.

**Reference run** (train on games 1 and 3, test on game 2), 95% cluster-bootstrap intervals:

| | GRU | logistic | gbdt | heuristic |
|---|---|---|---|---|
| Window PR-AUC | **0.569** | 0.506 | 0.500 | 0.392 |
| Brier | 0.128 | 0.098 | 0.051 | 0.095 |
| Episode precision | **0.706** [0.61, 0.81] | 0.631 [0.55, 0.73] | 0.402 [0.33, 0.47] | 0.400 [0.17, 0.67] |
| Episode recall | 0.590 | 0.672 | 0.869 | 0.066 |
| False alarms / 90 | **14.5** | 23.1 | 76.1 | 5.8 |
| Median warning | 1.22 s | 1.24 s | 2.92 s | 1.18 s |
| Early alarms | 1 | 4 | 11 | 0 |

Decision thresholds, fixed on the training matches: GRU 0.93, logistic 0.98, gradient boosting
0.60, heuristic 0.71.

**Where this falls short, plainly.** Median warning time is 1.22 s. That is enough to precede the
event but still short for a broadcast insight, and it is the direct cost of buying precision with a
high threshold. Recall is 0.59, so four in ten box entries pass unflagged. The false-alarm rate on
the held-out match (14.5 / 90) overshoots the 12 / 90 budget the threshold was set to on the
training matches. A threshold fitted on two matches does not transfer exactly to a third, which is
itself a useful measure of how little three matches support. The Brier score shows the GRU is the
*least* well calibrated of the four: it ranks well, but its probabilities should not be read as
frequencies. Improving warning time without giving back precision is the main open problem, and is
more likely to come from a longer horizon and more matches than from a larger model.

**A caveat on the intervals.** Bootstrap recall intervals are biased low. Resampling draws
possession clusters with replacement, so a cluster can appear several times while alarm-to-episode
matching stays one-to-one; duplicated episodes cannot all be matched. Precision and false-alarm
intervals are unaffected. Treat the fold-to-fold spread as the real measure of uncertainty: with
three matches, no interval can represent between-match variance.

## Latency

`make benchmark`, CPU only (Apple M-series, single ORT thread), batch size 1, which is the size the
live path actually uses:

| Runtime | p50 | p95 | p99 | Throughput | Cold start |
|---|---|---|---|---|---|
| PyTorch | 0.285 ms | 0.317 ms | 0.388 ms | 3,455 /s | 0.6 ms |
| ONNX Runtime | **0.047 ms** | **0.054 ms** | **0.065 ms** | 20,725 /s | 0.3 ms |

ONNX parity on 512 real windows: max absolute difference **1.19e-07**, zero ranking changes.
The check earns its place: it caught a stale export during development, when the benchmark reused
an `.onnx` file written before a retrain.
Standardisation and the sigmoid are folded into the exported graph, so the artifact is
self-contained.

Against the reference targets: p95 inference 0.054 ms against a 20 ms budget, and replay processes
a full 94-minute match through the engine in seconds, far faster than real time.

## Reliability

The system degrades to silence rather than fabricating output:

- an incomplete or invalid window emits **nothing**, and increments a specific suppression counter;
- no model loaded → `/ready` is false, `/model` returns 503, and suppressions are labelled
  `model_unavailable`, never confused with a quiet model;
- a model whose feature-schema hash disagrees with the running code is **refused**, not silently
  swapped for another;
- replay interruptions reset editorial state, so a stale insight cannot appear against new frames;
- the rule-based fallback is labelled `is_ml: false` in the API, the metrics and the demo UI.

Verified against deliberately degraded feeds. A full match under the `degraded` profile
(2,790 dropped frames, 1,397 duplicated, 1,368 out of order) runs clean and emits 20 insights,
about one every five minutes.

```bash
football-insights replay --match Sample_Game_2 --fault-profile degraded --seed 42
```

Fault injection is a pure function of `(profile, seed, frames)`: same seed, byte-identical stream.

## Example output

```
[ 1043.2s] Attacking threat is building: 3 attackers ahead of the ball,
           nearest defender 11 m away and 18 m of space ahead of the ball  (p=0.94)
[ 1298.7s] Signs of sustained pressure building: 2 recent penalty-area entries
           and 21 s of unbroken possession  (p=0.96)
```

Wording is template-based and deterministic. No language model is involved. Every headline is
hedged, and a test asserts that against a lexicon so an unhedged template cannot ship. The factual
clause is measured from the same window, so it can be stated plainly while the headline is not.

## The demo

`make demo` builds the React app into the package and serves it at
<http://127.0.0.1:8000>. It serves two readers at once.

The main view is the viewer's. The shaded penalty area is the one being attacked (the target of
the prediction), with a direction marker and a caption naming the side. The confidence panel
answers *"chance of a penalty-area entry"* as a position relative to the reporting line rather
than a percentage: the model is trained with a heavy positive-class weight, so it ranks well but
its output is not a frequency, and `docs/model_card.md` is explicit that it must not be shown to
an audience as one. Below it, a sparkline carries the last thirty seconds with breaks wherever
the model declined to score, because joining across a gap would draw a trend through frames that
were never scored. Each insight card shows the measured facts behind it as separate chips, so
every one can be checked against the pitch beside it. When nothing is being said, the panel says
why, held for four frames so the reason is readable rather than flickering at 12.5 Hz.

**Diagnostics** (a toggle in the header, remembered across reloads) opens a panel underneath with
the raw score, model metadata and schema state, the full fault summary, the malformed-event count,
and the distribution of editorial suppression reasons with the emit-to-suppress ratio. Those
suppression counts come from an exact server-side rollup published once per second of match time,
not from the frame stream: frames are published at half the rate they are scored, so a total
accumulated in the browser would be half the truth.

**Controls.** Pause/resume, four speed presets, and restart. The live speed is stated separately
because the service may be running at a value no preset offers: `make serve` uses 8x. The
progress bar is deliberately not draggable: there is no seek behind it, and a handle would be a
promise the service cannot keep. Reaching the end of a replay reports *finished*, not *offline*,
and offers a restart.

**Changing match.** The picker in the header switches which match is being replayed, in place,
without a reload. `GET /replay/matches` lists the three catalogued matches and whether their raw
files are actually on disk; one that has not been downloaded is shown and disabled rather than
hidden, so a build that knows about three matches does not present itself as having one. The
switch takes about two seconds — almost all of it parsing two 32 MB tracking files — so it runs
off the event loop and the transport says which match is loading while the pitch holds its last
frame. Every open tab follows the change: the server publishes it on the stream rather than each
client inferring it from its own click. The starting match is still `serve --match`.

**Pipeline controls** are **off by default**. With `serve --dev-tools` (or
`FI_SERVICE__ENABLE_PIPELINE_CONTROLS=1`) a panel appears below diagnostics that runs the five
Makefile stages — `data`, `prepare`, `train`, `evaluate`, `benchmark` — as tracked background
jobs, one at a time, with streamed output and a stop button. Each runs in its own process, so a
multi-minute training run does not hold the GIL against the replay loop and make the live pitch
stutter, and cancellation actually stops the work. A successful `train` reloads the predictor and
a successful `prepare` reloads the current match, because otherwise the process would keep serving
what it loaded at startup with nothing on screen saying so; a newly trained model whose feature
schema disagrees with the running build is still refused, and the old one kept.

It is off by default because the service has no authentication and mounts the demo at `/`: anyone
who could reach the port could otherwise start a 180 MB download and pin the CPU for minutes. That
is fine on a development machine and is not fine in the published container, so turning it on is a
deliberate act. When it is off the routes are not registered at all — they 404 like any other
absent path and never appear in the OpenAPI schema.

**Keyboard and screen readers.** Every control is reachable by Tab with a visible focus ring, and
Space toggles pause from the transport region. The pitch canvas carries a text alternative
updated about once a second; the insight feed is a polite live region that announces additions
only. Status is never carried by colour alone: a degraded fault count reads `2,790 · degraded`.

The demo is not optimised for small screens: below about 900 px it stacks to one column, but the
pitch is dense at phone widths and no mobile layout was attempted.

## Testing and CI

```bash
make test        # full suite, no network or dataset required
make check       # lint + mypy strict + pyright strict + tests
```

Both mypy and Pyright run in strict mode and both gate CI. The defects that the
typing and code-health pass surfaced, including several that had nothing to do
with types, are listed in [CHANGELOG.md](CHANGELOG.md).
[docs/code-quality.md](docs/code-quality.md) is the overview: what gates a
change, the typing policy behind it, and what is verified by hand against a real
browser and an installed wheel rather than by the gate.

Coverage of the things most likely to be wrong and hardest to notice:

- **Temporal leakage.** Features built at `t` are byte-identical when future events are deleted,
  mutated or fabricated, and when an in-flight event's outcome is tampered with.
- **Playing direction.** Sample game 3 declares its direction; the test withholds that and asserts
  the inference used on games 1 and 2 reproduces it exactly.
- **Determinism.** Identical seeds give identical fault streams and identical insight sequences.
- **Suppression.** Every reason in the enum, plus cooldown, staleness and reset-after-interruption.
- **ONNX parity**, API schema rejection, window validity, episode grouping sensitivity.

## Repository

| Path | |
|---|---|
| [`src/football_insights/data/`](src/football_insights/data/) | acquisition, both Metrica parsers, validation, orientation |
| [`features/causal.py`](src/football_insights/features/causal.py) | the forward-blind event view |
| [`labels/box_entry.py`](src/football_insights/labels/box_entry.py) | target and episode grouping |
| [`models/`](src/football_insights/models/) | baselines, GRU, evaluation, ONNX |
| [`insight/`](src/football_insights/insight/) | candidates, templates, editorial policy |
| [`serving/`](src/football_insights/serving/) | inference engine, FastAPI routes, shared state, the replay loop, match switching, pipeline jobs, metrics |
| [`docs/adr/`](docs/adr/) | why the load-bearing decisions were made |

## Verification matrix

Everything claimed above was executed on this machine, except one item that is named rather than
hidden.

| | |
|---|---|
| Tests, lint, type check | run locally |
| Data acquisition, preparation, orientation reports | run locally |
| Training, LOMO evaluation, bootstrap intervals | run locally |
| ONNX export, parity, latency benchmark | run locally |
| API, SSE stream, demo, degraded replay | run locally |
| Changing match on a running service | run locally, and timed: 2.05 s end to end |
| Pipeline job surface (start, stream, cancel, single-flight, gate on and off) | run locally through HTTP. The long stages were driven through the job machinery but **only `prepare` was started against real data, and it was cancelled after six seconds** rather than run to completion. |
| Demo layout after those changes | measured in Chrome at four viewports: confidence panel 290 px, document 900 px with twenty insights, both unchanged |
| **`docker build`** | **not run locally.** There is no Docker daemon on the development machine. The Dockerfile and compose file are written but unverified; CI builds them on push. |

## Limitations

- **Three matches.** Everything rests on 196 episodes. Fold-to-fold spread is wide and no
  confidence interval captures between-match variance. Nothing here supports a claim about
  generalising to another league, provider or season.
- **Warning time is short** (1.22 s median). Honest, and the main weakness.
- **Box entry is a proxy for danger**, not danger itself.
- **Possession is approximate**, derived causally from event annotation, so it lags real turnovers.
- **The GRU is poorly calibrated**, so it is usable for ranking, not as a probability.
- No automated retraining exists, and none is claimed. Drift monitoring produces a report.
- Anonymised sample data: no player identity, and nothing here predicts injury, intent or any
  personal characteristic.

Read [docs/model_card.md](docs/model_card.md) and [docs/data_card.md](docs/data_card.md) before
drawing conclusions from any number above.

## Licence and attribution

Code: MIT ([LICENSE](LICENSE)). Tracking and event data © Metrica Sports, from
[metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), downloaded at setup
time and never redistributed here. It carries **no formal licence**, only a request to be
responsible and acknowledge the source.
