/**
 * The viewer-facing panels: the page header, and what the model currently
 * thinks. Playback lives in `transport.jsx` and engineering detail in
 * `diagnostics.jsx`, so each reader's surface is one file.
 */

import {
  BAND_COPY,
  bandFor,
  readinessOf,
  runtimeSummary,
  suppressionLabel,
  trendOf,
} from './format.js'

/**
 * What this deployment actually is, in four lines.
 *
 * Every value comes from `/ready` rather than from copy written here, so the
 * page cannot claim something the running service is not doing. That matters
 * most for the two facts a viewer would otherwise assume wrongly: the hosted
 * replay is a fixture this project generates, and the model scoring it was
 * trained on that fixture rather than on the real matches the README reports.
 *
 * A compact block rather than a banner across the page. The disclosure has to
 * be present and legible, and it does not have to be the loudest thing on a
 * screen whose subject is the pitch.
 */
export function RuntimeStatus({ runtime }) {
  const summary = runtimeSummary(runtime)
  if (summary === null) return null
  return (
    <dl className="runtime-status" aria-label="What this deployment is running">
      <div>
        <dt>Mode</dt>
        <dd>{summary.mode}</dd>
      </div>
      <div>
        <dt>Data</dt>
        <dd className={summary.dataIsSynthetic ? 'warn' : undefined}>{summary.data}</dd>
      </div>
      <div>
        <dt>Predictor</dt>
        <dd className={summary.predictorIsMl === false ? 'warn' : undefined}>
          {summary.predictor}
          {summary.predictorName ? ` — ${summary.predictorName}` : ''}
        </dd>
      </div>
      <div>
        <dt>Replay</dt>
        <dd>{summary.replay}</dd>
      </div>
    </dl>
  )
}

/**
 * Which match is on, and a way to change it.
 *
 * In the header rather than beside the transport controls for two reasons. The
 * match is what the whole page is about, not a property of playback; and the
 * transport sits inside the two-column layout, whose insight feed is bounded
 * against the height of the column the transport is in.
 *
 * Unavailable matches are listed and disabled rather than hidden. A build that
 * knows about three matches and has downloaded one should say so, not present
 * itself as a one-match deployment.
 */
function MatchPicker({ matches, current, disabled, onSelect }) {
  if (!matches.length) return null
  return (
    <div className="match-picker">
      <label htmlFor="match-select">Match</label>
      <select
        id="match-select"
        value={current ?? ''}
        disabled={disabled}
        onChange={(event) => onSelect(event.target.value)}
      >
        {current && !matches.some((m) => m.id === current) && (
          <option value={current}>{current}</option>
        )}
        {matches.map((match) => (
          <option key={match.id} value={match.id} disabled={!match.available}>
            {match.id.replace(/_/g, ' ')}
            {match.available ? '' : ' — not downloaded'}
          </option>
        ))}
      </select>
    </div>
  )
}

/**
 * Which of the rotating fixtures is on, and what it is meant to look like.
 *
 * The public demo cycles three tactical archetypes and offers no way to choose
 * between them, so the picker above is absent and nothing else on the page
 * would say why the football just changed character. Both strings come from the
 * service's own catalogue rather than being written here, so a fixture added or
 * renamed in the generator cannot leave the page describing the wrong one.
 *
 * Rendered only once the name is known. A placeholder would put "Unknown" in
 * the header for the first moments of every load, which is worse than a header
 * that grows by one line a beat later.
 */
function CurrentFixture({ name, narrative }) {
  if (!name) return null
  return (
    <div className="current-fixture">
      <span className="fixture-name">{name}</span>
      {narrative ? <span className="fixture-narrative">{narrative}</span> : null}
    </div>
  )
}

export function Header({
  ready,
  readyReason,
  showDiagnostics,
  onToggleDiagnostics,
  matches,
  currentMatch,
  onSelectMatch,
  switching,
  fixtureName,
  fixtureNarrative,
}) {
  const { label, tone } = readinessOf(ready)
  return (
    <header>
      <div>
        <h1>Live Football Insight Engine</h1>
        <p className="subtitle">
          Predicting imminent penalty-area entries from tracking data, and only saying so when it is
          worth saying.
        </p>
      </div>
      <div className="header-controls">
        <CurrentFixture name={fixtureName} narrative={fixtureNarrative} />
        <MatchPicker
          matches={matches ?? []}
          current={currentMatch}
          disabled={Boolean(switching)}
          onSelect={onSelectMatch}
        />
        <button
          type="button"
          className="ghost"
          aria-expanded={showDiagnostics}
          aria-controls="diagnostics"
          onClick={onToggleDiagnostics}
        >
          {showDiagnostics ? 'Hide diagnostics' : 'Show diagnostics'}
        </button>
        <div className={`status-pill ${tone}`} title={readyReason ?? undefined}>
          {label}
        </div>
      </div>
    </header>
  )
}

const SPARK_W = 300
const SPARK_H = 36

/**
 * A hand-drawn trend line for the last half-minute of scoring.
 *
 * Decorative, and marked as such: the trend it shows is also stated in words
 * beside it, which is what a reader who cannot see the line gets. The line is
 * broken into segments wherever the model declined to score, because joining
 * across a gap would draw a trend through frames that were never scored at all.
 */
function Sparkline({ points, threshold }) {
  if (points.length < 2) return <div className="spark spark-empty" />

  const step = SPARK_W / Math.max(1, points.length - 1)
  const y = (p) => ((1 - p) * SPARK_H).toFixed(1)

  const segments = []
  let run = []
  points.forEach((p, index) => {
    if (p === null || p === undefined) {
      if (run.length > 1) segments.push(run)
      run = []
      return
    }
    run.push(`${(index * step).toFixed(1)},${y(p)}`)
  })
  if (run.length > 1) segments.push(run)

  return (
    <svg
      className="spark"
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      focusable="false"
    >
      <line
        className="spark-threshold"
        x1="0"
        x2={SPARK_W}
        y1={y(threshold)}
        y2={y(threshold)}
        vectorEffect="non-scaling-stroke"
      />
      {segments.map((segment) => (
        <polyline
          key={segment[0]}
          className="spark-line"
          points={segment.join(' ')}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  )
}

/**
 * How likely the target box is to be entered, expressed as a band.
 *
 * Deliberately not a percentage. The model card is explicit that the raw
 * probability must not be shown to an audience as one — it is trained with a
 * heavy positive-class weight, so it ranks well but its numbers are not
 * frequencies. The bar and the threshold marker stay, because comparing an
 * estimate against the operating point is exactly what the service itself does.
 * The number itself lives in diagnostics, labelled for what it is.
 *
 * Every state renders the same slots. The panel used to be built from whatever
 * happened to be true at the time, so it changed height as the estimate moved:
 * measured at 246 px with nothing held, 272 px once a suppression reason stuck,
 * and 290 px above the reporting line, where the longer copy takes a second
 * line. That is 44 px of movement in the column beside the pitch, and it landed
 * on the pitch and the transport controls. The slots below reserve the tallest
 * copy each one can hold instead, so the numbers no longer move the layout.
 *
 * `isMl` is deliberately three-valued: `null` means the service has not
 * answered yet. Collapsing that into `false` showed every deployment the
 * fallback badge for as long as `/model` was in flight — 280 px settling to
 * 246 px on a cold load, which is the one jump every reader saw.
 */
export function Confidence({ probability, threshold, history, isMl, reason, horizonS }) {
  const band = bandFor(probability, threshold)
  const copy = BAND_COPY[band]
  const trend = trendOf(history)
  const held = suppressionLabel(reason)

  const valueAria = band === 'unscored' ? {} : { 'aria-valuenow': Math.round(probability * 100) }

  return (
    <section className="confidence" aria-labelledby="confidence-heading">
      <h2 id="confidence-heading">Chance of a penalty-area entry</h2>
      <p className="panel-note confidence-lede">
        How likely the shaded box is entered in the next{' '}
        {horizonS ? `${horizonS} seconds` : 'few seconds'}.
      </p>

      <div className={`band band-${band}`}>{copy.chip}</div>

      <div
        className="meter"
        role="meter"
        aria-labelledby="confidence-heading"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuetext={copy.chip}
        {...valueAria}
      >
        <div
          className={`meter-fill ${band === 'reporting' ? 'over' : ''}`}
          style={{ width: `${band === 'unscored' ? 0 : Math.round(probability * 100)}%` }}
        />
        <div className="meter-threshold" style={{ left: `${threshold * 100}%` }} />
      </div>

      <p className="panel-note confidence-detail">{copy.detail}</p>

      <Sparkline points={history} threshold={threshold} />
      <p className="panel-note confidence-trend">
        {trend ? `Trend, last 30 s: ${trend}` : 'Trend, last 30 s: not enough data yet'}
      </p>

      {/* The two things here come and go, so the region reserves room for them
          rather than appearing and disappearing under the pitch. */}
      <div className="confidence-foot">
        {reason && (
          <p className="reason">
            Currently quiet: <strong>{held}</strong>
          </p>
        )}

        {isMl === false && (
          <div className="fallback-badge" title="Rule-based fallback, not a trained model">
            Fallback — not ML
          </div>
        )}
      </div>
    </section>
  )
}

/** How long ago the replay panel last heard from the service. */
