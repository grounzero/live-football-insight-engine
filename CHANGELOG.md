# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Typing and code-health hardening, plus the defects that pass surfaced, all
listed below. No feature semantics, model contracts, evaluation behaviour or API
schemas changed. [docs/code-quality.md](docs/code-quality.md) describes how to
run the checks.

### Fixed
- **Changing the replay speed hung the stream.** `ReplayPlayer.stream` keeps its
  pacing anchor in local variables, so `set_speed`, arriving on another task,
  could not reset it, despite its docstring saying it did. Lowering the speed
  made the loop sleep off the match time already replayed at the old rate: 21 s
  of silence after only 3 s of playback going 8x -> 1x, growing linearly with
  playback time, and unrecoverable without restarting the service. Since the
  demo starts at `--speed 8`, the 1x, 2x and 5x buttons all triggered it.
- **`src/football_insights/data/` was excluded from the repository.** An
  unanchored `data/` rule in `.gitignore`, meant for the downloaded dataset at
  the repository root, also matched the ingestion package. A clean clone of
  0.1.0 could not import `football_insights.cli`. The rules are now anchored
  (`/data/`, `/artifacts/`).
- Ruff respects `.gitignore`, so those eight modules had never been linted.
  Fifteen findings in them are resolved.
- A dead `hasattr(SPEC, "sequence_length")` branch in the editorial tests;
  `FeatureSpec` has no such attribute, so the fallback always won.
- **The built demo was missing from the wheel and sdist.** Only `py.typed` was
  declared as package data, so an installed copy found no `index.html`,
  `_mount_demo` took its soft-skip path and the service silently served the API
  only, so the demo the README points at never appeared. `serving/static` is now
  declared as package data and the mount is verified from an installed wheel.
- `make audit` no longer swallows its exit status with `|| true`, and skips the
  editable project instead of failing to find it on PyPI.
- **A finished replay left every browser tab in a reconnect loop.** The stream's
  `end` event was unhandled, so the client never closed its `EventSource`. The
  server had already returned from its publisher and closed the response, the
  browser reconnected a few seconds later, the service started a fresh replay
  task whose player sat at the last frame, and it ended again, for as long as
  the tab stayed open. A clean finish also presented as `offline`, identical to
  a crashed backend.
- **Rewinding the replay made it sprint.** `ReplayPlayer.reset` rewound the
  position but left the pacing anchor measured from a match time the replay was
  now far behind, so every frame looked overdue and was emitted with no delay
  until it caught back up. The same class of defect as the `set_speed` hang
  above, arriving from the other direction, and the reason `reset` could not
  simply be exposed as-is. The loop's own compensating `_drop_anchor` call is
  gone; `reset` owns the anchor.
- **Restarting could have left the engine scoring nothing.** A restart request
  can land while the replay loop is holding a frame taken from the old position.
  Letting that frame through leaves the engine's monotonic frame check ahead of
  everything about to be replayed, so every frame of the new run is rejected as
  out of order, while the pitch keeps animating, because frames are published
  whether or not the engine accepted them. The handler now requests a restart
  and the loop applies it, discarding the held frame.
- **The demo contradicted its own model card.** `docs/model_card.md` states that
  the raw probability must not be surfaced to an audience as a percentage: the
  model is trained with a ~16:1 positive class weight, so it ranks well but its
  output is not a frequency. Both the confidence panel and every insight card
  displayed one. The viewer surface now shows which side of the reporting line
  the estimate falls on; the number itself moved to diagnostics, labelled as a
  ranking score.
- **The readiness pill rendered red on every cold load.** Its label had three
  states and its styling two, so the "checking" that precedes the first `/ready`
  answer was painted as a failure. `/model` and `/ready` were also fetched once
  at mount and never again, so a service that became ready a second after load
  was described as broken until someone reloaded.
- One failed `/replay/status` poll blanked the whole replay panel. The last good
  reading is now kept, and after three consecutive failures the panel says how
  stale it is instead of claiming to know nothing.
- `insight.facts[]`, the structured `{key, text, value}` evidence behind each
  insight, was discarded in favour of the pre-joined sentence, and the
  `GET /insights` history endpoint was never called, so a refresh lost the
  timeline the service was already keeping.
- The pitch canvas never resized. Mid-playback the next frame corrected it 80 ms
  later; paused or finished it stayed stretched indefinitely, which is exactly
  when someone resizes to look more closely.
- The favicon would not have shipped: `public/` passthrough lands at the root of
  the build output, which neither `static/index.html` nor `static/assets/*`
  matches, so the packaged page would have requested an icon the wheel did not
  contain.
- **The confidence panel changed height as the estimate moved**, and the pitch
  and transport controls beside it moved with it. Measured at 1440×900: 246 px
  with nothing held, 272 px once a suppression reason stuck, 290 px above the
  reporting line where the longer copy wraps. A cold load added a fourth
  height, because `isMl` collapsed "the service has not answered yet" into "this
  is not a trained model", so every deployment showed the fallback badge until
  `/model` arrived, 280 px settling to 246 px. The slots now reserve the tallest
  copy they can hold and `isMl` is three-valued: 290 px in every state.
- **Insight history grew the page without limit.** The feed renders up to twenty
  insights and nothing bounded them: twenty measured 2,092 px of list, a
  2,567 px document and the diagnostics panel 2,525 px below its toggle. The
  list body now scrolls inside a panel bounded by `--insight-body-max`, so the
  document stops at 900 px with twenty insights and diagnostics sits 844 px
  down. Ordering, the `aria-live` region and the twenty-insight cap are
  unchanged; the heading, count and finished notice stay outside the scroll.

### Added
- **The match can be changed from the demo.** It was fixed at process start:
  `serve --match` built one engine and one `ReplayPlayer`, and the UI only
  displayed the id in diagnostics. A picker in the header now switches it in
  place, in about 1.8 s — measured 1.44–1.56 s parsing the two 32 MB tracking
  files, 0.31–0.36 s materialising 145k frames — run in a worker thread so the
  event loop is never held for it. Two `GET /replay/matches` and
  `POST /replay/match`, and a sixth SSE message type, `match`, so every open tab
  follows a change rather than only the tab that asked for it.

  Two parts of this were not obvious. The replay loop binds its player into a
  local and then sits inside its stream, so assigning a new one does nothing —
  the task has to be cancelled and awaited. And cancelling it published the
  `end` marker, which clients close their stream on for good, so a change of
  match would have presented itself as the end of the match and disconnected
  every tab. The predictor and the metric registry are carried across the swap:
  a fresh `Metrics` would build a registry `GET /metrics` never reads, freezing
  every counter from the moment someone changed match.
- **The pipeline stages can be run from the demo**, behind
  `serve --dev-tools` (`FI_SERVICE__ENABLE_PIPELINE_CONTROLS`), **off by
  default**. `data`, `prepare`, `train`, `evaluate` and `benchmark` run as
  tracked jobs under `/jobs` with streamed logs, one at a time, cancellable.
  Nothing shells out: each Make target is a thin wrapper over one library call,
  so the jobs call those. Each runs in its own spawned process — in a thread it
  would hold the GIL against the replay loop and the live pitch would stutter
  for the whole run, and a pool cannot cancel a task once started, which is no
  use for a job whose first act is a 180 MB download. A successful `train`
  reloads the predictor and a successful `prepare` reloads the match, so the
  service stops silently serving what it loaded at startup; a schema mismatch is
  still refused and the working model kept.

  Off by default because the service has no authentication and mounts the demo
  at `/`. When disabled the routes are not registered at all, so they 404 and
  never appear in the published schema.
- **The demo shows what it is predicting.** The attacking team and its direction
  now travel on the SSE frame payload (`attacking_team`, `attacking_right`), so
  the target penalty area is shaded and a direction marker is drawn. The code to
  draw that box had been written and correct since the first release, but the
  browser was never told which end to shade, so a screen describing a forecast
  never showed what was being forecast.
- **The editorial layer is visible.** Each frame carries the current
  `suppression` reason, and an exact per-reason rollup is published once per
  second of match time. It is aggregated on the server because frames are
  published at half the rate they are scored: totals derived in the browser
  would be a sample presented as a total.
- **Replay restart**, as `{"restart": true}` on the existing
  `POST /replay/control`. No new route, so the published route set is unchanged.
- `horizon_s` on `GET /model`, so the demo can name the prediction window
  without hardcoding a number that would drift if the window were retuned.
- A diagnostics panel below the main layout holding raw score, model metadata,
  suppression distribution, emit-to-suppress ratio, malformed-event count and
  the full fault summary, all six fields rather than just `dropped`.
- Keyboard and screen-reader support the demo previously had none of: the pitch
  canvas has a text alternative, the insight feed is a polite live region,
  speed buttons carry `aria-pressed` and spoken labels, every control has a
  visible focus ring, and Space toggles pause.
- `npm run test:ssr`, twelve server-rendered structural assertions over the
  viewer surface (live-region semantics, the scrolling body, the confidence
  panel's slots in every state, no audience-facing percentage, diagnostics
  outside `main`), in the demo CI job. No new dependency: `react-dom/server`
  ships with react-dom and vite builds the JSX for `node --test`.
- Strict Pyright across `src` and `tests`, in `make check` and in CI.
- Partial local stubs for scikit-learn and onnxruntime under `stubs/`.
- `scripts/codescene.sh` and `make codehealth` for local CodeScene analysis.
- dtype-precise NumPy array aliases in `football_insights.types`.
- 35 tests covering the YAML and EPTS JSON parsing boundaries, the API route
  surface, replay pacing under speed and pause control, and the synthetic
  generator's seed-to-output contract (pinned digests, so a reordered RNG draw
  cannot pass silently).

### Security
- `pytest` raised to `>=9.0.3`. 8.x is affected by PYSEC-2026-1845 and the
  previous `<9` cap made the fixed release uninstallable. `pip-audit` is now
  clean.

### Changed
- FastAPI routes moved from closures in one 148-line function to four
  module-level routers; `AppState` now owns its replay task.
- `serving/app.py` split along the lines the match-switch work exposed:
  `state.py` (shared state and the replay task's lifetime), `stream.py` (the
  replay loop and every message it publishes), `switching.py` (rearranging a
  running service), `loader.py` (building the pieces, needed at runtime as well
  as at startup, which `bootstrap.py` could not provide without an import
  cycle). CodeScene: app.py 9.44 -> 9.68, and the extracted modules score 10.00,
  10.00, 9.68 and 9.68.
- The demo split the same way: `transport.jsx` out of `panels.jsx`, and
  `jobs.js` out of `hooks.js`. Every demo module now scores 10.00, including
  `panels.jsx`, which was 9.38.
- Demo build tooling to vite 8 and `@vitejs/plugin-react` 6 (both major
  releases; Node floor is now 20.19 / 22.12, which CI already meets).
- uvicorn to 0.52.1.
- The React demo split into `hooks.js` and `panels.jsx`; `App.jsx` is now
  composition only.
- `generate_synthetic_match` split into named stages behind a `_PeriodClock`
  value; output verified byte-identical across nine configurations.
- The four ingestion modules restructured for maintainability:
  `orientation.py` 6.47 -> 9.68, `metrica_epts.py` 7.73 -> 10.00,
  `metrica_csv.py` 8.47 -> 9.68, `validate.py` 8.83 -> 9.38. Output verified
  identical against all three real Metrica matches, covering every tracking
  array, event, orientation decision and validation finding.

## [0.1.0] - 2026-08-01

First working system, end to end.

### Added
- Ingestion for both Metrica formats: CSV (games 1–2) and EPTS/FIFA XML + JSON (game 3), including
  the eleven `DataFormatSpecification` layouts game 3 uses to re-map columns at substitutions.
- Validation that fails loudly on unordered frames, backwards timestamps, implausible coordinates
  and unalignable events, and reports non-fatal data-quality findings.
- Attacking-direction inference from four tiers of evidence with an audit artifact, structural
  checks, and a documented override path.
- `CausalEventView`: forward-blind event access that hides in-flight event outcomes by construction.
- Penalty-area-entry labelling with configurable window, horizon and stride, plus episode grouping.
- 39 identity-invariant features with a versioned schema hash the serving path enforces.
- Baselines (logistic regression, gradient boosting), a compact GRU, and a labelled rule-based
  fallback measured separately.
- Evaluation at window and episode level with cluster-bootstrap intervals over possession sequences.
- ONNX export with standardisation folded into the graph, parity checking and latency benchmarking.
- Deterministic replay fault injection (`clean`, `jitter`, `degraded`, `hostile`).
- FastAPI service with health, readiness, model metadata, prediction, SSE stream and Prometheus
  metrics in two namespaces; React + canvas demo.
- Drift and data-quality reporting.
- 115 tests, none requiring network or the dataset.

### Known limitations
- Three matches, 196 episodes. Wide fold-to-fold spread; no interval captures between-match variance.
- Median warning time ~0.6 s at the chosen operating point.
- The GRU is the least well-calibrated of the four models; usable for ranking, not as a probability.
- Bootstrap recall intervals are biased low under one-to-one alarm matching.
- `docker build` has not been executed locally (no daemon on the development machine).
