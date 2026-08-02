import {
  ABSENT,
  STREAM_COPY,
  describe,
  formatCount,
  readinessOf,
  suppressionLabel,
} from './format.js'

/**
 * A labelled value list.
 *
 * `note` carries a word for whatever the colour is saying, because a tone alone
 * is not information: "2,790" in amber and "2,790" in grey are the same string
 * to anyone who cannot separate the two.
 */
function Rows({ rows }) {
  return (
    <dl>
      {rows.map(({ label, value, tone, note }) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd className={tone ?? ''}>
            {value}
            {note ? <span className="qualifier"> · {note}</span> : null}
          </dd>
        </div>
      ))}
    </dl>
  )
}

/**
 * Where the editorial layer's decisions actually went.
 *
 * Built from the service's own rollups, which count every reviewed frame.
 * Deriving this from the frame stream instead would halve every number, because
 * frames are published at half the rate they are scored — and a halved total
 * presented as a total is worse than no total at all.
 */
function SuppressionBars({ suppression }) {
  const entries = Object.entries(suppression.counts).sort((a, b) => b[1] - a[1])
  const suppressed = entries.reduce((sum, [, value]) => sum + value, 0)

  if (!entries.length) {
    return <p className="empty">No decisions reviewed yet.</p>
  }

  return (
    <ul className="bars">
      {entries.map(([reason, count]) => {
        const share = Math.round((count / suppressed) * 100)
        return (
          <li key={reason}>
            <div className="bar-head">
              {/* An unrecognised reason is shown verbatim rather than hidden: a
                  value this build has not been taught about is exactly what an
                  engineer needs to see. */}
              <span>{suppressionLabel(reason)}</span>
              <span className="bar-value">
                {formatCount(count)} · {share}%
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${share}%` }} />
            </div>
          </li>
        )
      })}
    </ul>
  )
}

function serviceRows({ ready, readyReason, stream }) {
  const readiness = readinessOf(ready)
  const streamCopy = STREAM_COPY[stream.status] ?? STREAM_COPY.connecting
  return [
    {
      label: 'Readiness',
      value: readiness.label,
      tone: readiness.tone,
      note: readyReason ?? '',
    },
    {
      label: 'Stream',
      value: streamCopy.label,
      tone: streamCopy.tone,
      note: streamCopy.note,
    },
    { label: 'Frames received', value: formatCount(stream.framesSeen) },
    {
      label: 'Malformed events',
      value: formatCount(stream.malformed),
      tone: stream.malformed > 0 ? 'bad' : '',
      note: stream.malformed > 0 ? 'transport defect' : '',
    },
  ]
}

function modelRows({ model, probability, threshold }) {
  return [
    { label: 'Model', value: describe(model, (m) => `${m.name} ${m.version}`) },
    { label: 'Kind', value: model?.kind ?? ABSENT },
    {
      label: 'Trained on ML data',
      value: model ? String(model.is_ml) : ABSENT,
      tone: model && !model.is_ml ? 'warn' : '',
      note: model && !model.is_ml ? 'rule-based fallback' : '',
    },
    {
      label: 'Raw score',
      value: probability === null ? ABSENT : probability.toFixed(4),
      note: 'ranking score, not a calibrated probability',
    },
    { label: 'Decision threshold', value: threshold.toFixed(4) },
    { label: 'Horizon', value: model?.horizon_s ? `${model.horizon_s}s` : ABSENT },
    {
      label: 'Feature schema',
      value: describe(model?.feature_schema_hash, (h) => h.slice(0, 8)),
      // Only a positive mismatch is an error; "unknown" is not.
      tone: model?.schema_matches === false ? 'bad' : '',
      note: model?.schema_matches === false ? 'mismatch' : '',
    },
    {
      label: 'Running schema',
      value: describe(model?.running_feature_schema, (h) => h.slice(0, 8)),
    },
    { label: 'Trained at', value: model?.trained_at ?? ABSENT },
  ]
}

function replayRows({ replay, stale, lastSeen }) {
  const faults = replay?.fault_summary ?? {}
  const dropped = faults.dropped ?? 0
  const age = lastSeen ? `${Math.round((Date.now() - lastSeen) / 1000)}s ago` : 'never'
  return [
    {
      label: 'Status',
      value: describe(replay, (r) => (r.paused ? 'paused' : r.running ? 'running' : 'stopped')),
      tone: stale ? 'warn' : '',
      note: stale ? `stale, last seen ${age}` : '',
    },
    { label: 'Match', value: replay?.match_id ?? ABSENT },
    { label: 'Speed', value: describe(replay, (r) => `${r.speed}×`) },
    { label: 'Position', value: describe(replay, (r) => `${r.frames_emitted} / ${r.total_frames}`) },
    { label: 'Fault profile', value: describe(replay, (r) => `${r.fault_profile} · seed ${r.seed}`) },
    { label: 'Source frames', value: formatCount(faults.source_frames ?? 0) },
    { label: 'Emitted frames', value: formatCount(faults.emitted_frames ?? 0) },
    {
      label: 'Dropped',
      value: formatCount(dropped),
      tone: dropped > 0 ? 'warn' : '',
      note: dropped > 0 ? 'degraded' : '',
    },
    { label: 'Duplicated', value: formatCount(faults.duplicated ?? 0) },
    { label: 'Delayed', value: formatCount(faults.delayed ?? 0) },
    { label: 'Reordered', value: formatCount(faults.reordered ?? 0) },
  ]
}

function editorialRows({ stream, stickyReason }) {
  const { suppression } = stream
  const suppressed = Object.values(suppression.counts).reduce((sum, v) => sum + v, 0)
  const ratio = suppression.emitted > 0 ? Math.round(suppressed / suppression.emitted) : null
  const latest = stream.insights[0]
  return [
    { label: 'Frames reviewed', value: formatCount(suppression.frames) },
    { label: 'Insights emitted', value: formatCount(suppression.emitted) },
    { label: 'Suppressed', value: formatCount(suppressed) },
    {
      label: 'Emit-to-suppress',
      value: ratio === null ? ABSENT : `1 in ${formatCount(ratio + 1)}`,
    },
    { label: 'Reason now (raw)', value: suppressionLabel(stream.reason) },
    { label: 'Reason now (held)', value: suppressionLabel(stickyReason) },
    { label: 'Latest detail', value: latest?.detail || ABSENT },
  ]
}

/**
 * Engineering detail, below the main layout rather than beside or over it.
 *
 * A third column would halve the pitch at the page's maximum width and force a
 * canvas re-layout on every toggle; an overlay would need a hand-rolled focus
 * trap and would cover the live pitch the reader is trying to diagnose. Full
 * width underneath costs one scroll and keeps both audiences on screen at once.
 * Each card is a `<details>`, which is keyboard- and screen-reader-operable
 * without a line of ARIA.
 */
export function DiagnosticsPanel(props) {
  return (
    <section className="diagnostics" id="diagnostics" aria-labelledby="diagnostics-heading">
      <h2 id="diagnostics-heading">Diagnostics</h2>
      <div className="diag-grid">
        <details open>
          <summary>Service</summary>
          <Rows rows={serviceRows(props)} />
        </details>
        <details open>
          <summary>Model</summary>
          <Rows rows={modelRows(props)} />
        </details>
        <details open>
          <summary>Replay</summary>
          <Rows rows={replayRows(props)} />
        </details>
        <details open>
          <summary>Editorial</summary>
          <Rows rows={editorialRows(props)} />
          <SuppressionBars suppression={props.stream.suppression} />
        </details>
      </div>
    </section>
  )
}
