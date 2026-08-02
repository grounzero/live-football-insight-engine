# Code quality and assurance

## Summary

This is an overview of how the project is kept honest: what gates every change,
what is checked by hand, and what is deliberately left outside the gate.

It records the *approach*, not a scorecard. Counts and scores move with every
commit, so pinning them in a document only guarantees the document goes stale.
Where a number matters it belongs in the tool that produced it: the test runner
prints its own totals, `CHANGELOG.md` records what changed in a release, and
CodeScene reports live in `artifacts/code-quality/`.

The shape of it: two type checkers and a linter gate every change, the test
suite runs without network or data, dependency and code-health analysis run
alongside rather than in the way, and the parts that cannot be asserted from
static analysis (layout, accessibility, packaging) are verified against a real
browser and a real installed wheel.

## Quality gates

```bash
make check     # ruff, mypy strict, pyright strict, pytest
```

`make check` is the gate. It needs no credentials, no network and no dataset:
every test runs against a seeded synthetic match generated in-process, so a
fresh clone can verify itself before downloading anything.

Individually:

```bash
make lint       # ruff check src tests
make format     # ruff format + --fix
make typecheck  # mypy (strict)
make pyright    # pyright (strict)
make test       # pytest
make audit      # pip-audit
```

CI runs the same checks across a Python version matrix, plus jobs that are
impractical locally: an ONNX export-and-parity smoke test, a service smoke test,
the demo build with its server-rendered structural tests, a dependency audit,
and the container build.

Two markers keep the suite usable. `requires_data` covers the tests that need
the downloaded Metrica dataset; they skip with a message when it is absent, so
`make test` stays green on a clean clone while CI runs `-m "not requires_data"`
explicitly. `slow` marks the tests worth excluding from a tight edit loop.

`filterwarnings` promotes this project's own deprecation warnings to errors, so
an internal API cannot be quietly deprecated and left in use.

## Type checking policy

Both mypy and Pyright run in strict mode and both must pass. They disagree often
enough to be worth running together: mypy is stricter about untyped definitions,
Pyright about partially-unknown types leaking out of third-party libraries.

Pyright is configured entirely in `pyrightconfig.json`, scoped to `src` and
`tests`. Two settings there are load-bearing:

- `venvPath`/`venv`, without which Pyright resolves imports against the wrong
  interpreter and reports thousands of phantom errors. CI installs into `.venv`
  for exactly this reason.
- `pythonVersion`, because numpy's bundled stubs use syntax that is not parsed
  at the project's minimum runtime version. This sets the analysis level only;
  the package still runs on the floor declared in `pyproject.toml`.

The rules that keep it strict, rather than strict-looking:

- **No file-wide suppressions**, and no excluding a source directory to make it
  pass.
- **`Any` belongs at boundaries only**: JSON, YAML, environment variables,
  third-party results. Parse and validate into concrete types at the edge;
  everything inland is precisely typed.
- **Prefer a typed adapter to a repeated ignore.** Where a library is
  under-annotated, wrap it once rather than suppressing at every call site.
  `models/_torch_typing.py` and `tests/support.py` exist for this.
- **Every retained ignore names its diagnostic and its reason**, as
  `# pyright: ignore[reportUnknownMemberType]` with a comment identifying the
  upstream signature at fault. A bare `# type: ignore` does not pass review.
- For a library with no `py.typed`, add a partial stub under `stubs/` covering
  only what is used. See `stubs/README.md`.

### Third-party typing limitations

These are upstream gaps, not project decisions. They are why the local stubs and
the retained suppressions exist.

| Library                | Gap                                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| scikit-learn           | no `py.typed`, no maintained stub distribution; partial local stubs                                       |
| onnxruntime            | no `py.typed`; partial local stub                                                                         |
| torch                  | `from_numpy`, `manual_seed`, `Optimizer.step`, `onnx.export` have unannotated parameters                  |
| starlette `TestClient` | annotated against httpx's private `_types`; starlette is deprecating this integration in favour of httpx2 |
| pytest                 | ships `py.typed` but leaves `approx` unannotated                                                          |
| numpy                  | `__eq__` is `Any` by necessity; stubs need a recent `pythonVersion`                                       |

No licensing constraints apply to the local stubs: they are original
descriptions of public API signatures, not copied source.

## CodeScene

Code health is a supplementary check, deliberately *not* part of `make check`:
the CLI requires a personal access token, and no contributor should need
proprietary credentials to submit a change.

```bash
make codehealth                # delta against the merge base with main
scripts/codescene.sh --review  # also review each changed file
scripts/codescene.sh --all     # review every tracked source file
```

The CLI analyses JS and JSX as well as Python, so the demo is worth checking
directly. The wrapper excludes the built bundle under `serving/static/assets/`:
CodeScene will happily score minified output, and the result is both meaningless
and large enough to look like a real regression in a summary.

One behaviour is worth knowing: **`cs` exits 0 even when unauthenticated**,
printing a "set up a Personal Access Token" message instead of analysing
anything. A naive wrapper would report success having checked nothing, so
`scripts/codescene.sh` probes for that message and fails explicitly. The token is
never echoed, logged or written to a report. Reports go to
`artifacts/code-quality/`, which is git-ignored.

Findings are evaluated, not blindly satisfied. Vectorised numerical code, the
feature builder and the label definition carry genuine domain complexity, and a
finding against them is often worth retaining with a note in the pull request
rather than refactoring away. Do not relax a rule to improve a score.

### The match-switching and pipeline pass

Adding a match picker and the gated job surface nearly doubled `serving/app.py`,
and CodeScene reported the cost immediately: 9.44 to 8.28, with a new **Low
Cohesion** finding — one file holding shared state, the replay loop, the routing
layer and the swap machinery. The response was to split it along the seams the
new work exposed rather than to accept the score:

| Module | Score | Holds |
| --- | --- | --- |
| `serving/app.py` | 9.68 | Request/response models, routers, `create_app` |
| `serving/state.py` | 10.00 | State shared across requests; the replay task's lifetime |
| `serving/stream.py` | 9.68 | The replay loop and every message it publishes |
| `serving/switching.py` | 10.00 | Changing match, and reacting to a finished job |
| `serving/loader.py` | 9.68 | Building predictor, match, engine and player |
| `serving/jobs.py` | 10.00 | The stage registry, worker process and job routes |

The demo split the same way, and for the same reason — `panels.jsx` had become a
drawer holding both viewer-facing panels and playback controls:

| Module | Before | After |
| --- | --- | --- |
| `demo/src/hooks.js` | 10.00 | 10.00 (job hooks moved to `jobs.js`) |
| `demo/src/panels.jsx` | 9.38 | 10.00 (`Transport` moved to `transport.jsx`) |
| `demo/src/App.jsx` | 10.00 | 10.00 |
| `demo/src/transport.jsx` | — | 10.00 |
| `demo/src/jobs.js`, `demo/src/pipeline.jsx` | — | 10.00 |

Two findings were left in place after review. `PredictRequest._rectangular_and_finite`
(cc = 10) is a flat sequence of independent input checks, each with its own
message, and folding them together would make a 422 less useful. `_publish_results`
in `stream.py` takes five arguments because one processed frame genuinely has
three publication cadences; bundling them into a value object to satisfy the
threshold would add a type that means nothing to a reader.

## Assurance beyond the gates

Static analysis and unit tests do not catch everything, and the checks below
exist because each of them has caught something the gate could not. The specific
defects are recorded in `CHANGELOG.md`; the point here is the class of problem
each check covers.

- **Concurrency and lifecycle.** Replay pacing, pause, speed changes and restart
  involve a control request arriving on a different task from the loop it
  affects. Type checkers see nothing wrong with a stale anchor held in a local
  variable. These are covered by tests that assert timing behaviour directly.
- **Packaging.** A package that imports cleanly from the source tree can still
  ship without its data files. This is only visible from an installed artifact.
- **Ignore rules.** An over-broad `.gitignore` pattern can silently exclude a
  source package, and linters that honour `.gitignore` will then skip it too, so
  the omission hides itself. Ignore rules are verified with `git check-ignore`
  against both source paths and runtime paths.
- **Editorial and presentation contracts.** The model card forbids showing the
  raw probability to an audience as a percentage. That is a documented constraint
  no type checker can enforce, so it is asserted in tests instead.

## Browser and accessibility verification

The demo's server-rendered tests (`npm run test:ssr`, run in CI) assert
*structure*: live-region semantics on the insight feed, the scrolling history
body, the confidence panel rendering the same slots in every state, diagnostics
sitting outside `main`, and no audience-facing percentage anywhere in the viewer
markup.

Structure is all that markup can prove. Geometry is a question about a rendered
page, so layout stability, focus visibility, keyboard reachability, reduced
motion, stream lifecycle and offline recovery are verified by driving a real
browser against the real service. That verification is deliberately manual and
not part of CI: it needs a browser, a running backend and, for the useful cases,
a real match to replay.

What it covers: the confidence panel holding one height across every scoring
state and viewport, the insight history staying bounded so the page does not grow
without limit, the pitch canvas not resizing as history accumulates, a clean
finish reporting *finished* rather than *offline* and closing its stream instead
of reconnecting forever, restart behaviour, recovery after the backend
disappears and returns, and that nothing is clipped or overflowing at high zoom.

## Packaging and installed-wheel verification

The wheel and sdist are built and then verified from a clean virtual environment
outside the source tree, because the source tree hides exactly the failures that
matter: imports resolve from the working directory, and data files are present
whether or not they were declared.

What is checked on the installed artifact: the runtime packages and `py.typed`
are present, the built demo ships with it, the served page references assets that
actually exist in the wheel, and the API, the JavaScript, the CSS and the favicon
all respond. What must *not* be present: raw or processed data, generated
artifacts, model files, experiment tracking, virtualenvs, `node_modules`,
code-health reports, or any absolute local path.

Asset filenames are content-hashed by the bundler and change on every build, so
package data is declared by glob and the served page is checked against the
wheel's actual contents rather than a remembered filename.

## Known limitations

- **Code health is not gated.** CodeScene needs credentials, so a change can
  merge without it having run. This is a deliberate trade: an external,
  credentialed service should not stand between a contributor and a pull request.
- **Browser verification is manual.** It is real, but it is not automated and
  will not catch a regression on its own. The SSR tests exist to cover the part
  of it that can be asserted from markup.
- **`docker build` is not run locally.** There is no Docker daemon on the
  development machine; CI is the only thing that builds the image.
- **The dependency audit is advisory in CI** and gating locally. A new advisory
  against a transitive dependency should surface without blocking an unrelated
  pull request, but `make audit` still fails on findings.
- **Evaluation rests on three matches.** No amount of code quality changes that,
  and the model and data cards state the consequences plainly.
