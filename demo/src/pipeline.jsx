import { useEffect, useRef, useState } from 'react'

import { useJobs } from './jobs.js'

/**
 * Tone per job state. `cancelled` is deliberately neutral: stopping a job you
 * started is an ordinary act, not a failure to flag in red.
 */
const STATE_TONE = {
  running: 'warn',
  succeeded: 'ok',
  failed: 'bad',
  cancelled: '',
}

/** Stages that change what the running service should be serving. */
const RELOADS = {
  prepare: 'reloads the current match',
  train: 'reloads the model',
}

function seconds(from, to) {
  const start = Date.parse(from)
  const end = to ? Date.parse(to) : Date.now()
  if (Number.isNaN(start) || Number.isNaN(end)) return null
  return Math.max(0, Math.round((end - start) / 1000))
}

function duration(record) {
  const value = seconds(record.started_at, record.finished_at)
  if (value === null) return ''
  if (value < 60) return `${value}s`
  return `${Math.floor(value / 60)}m ${String(value % 60).padStart(2, '0')}s`
}

/**
 * One run: what it was, how it ended, how long it took.
 *
 * Selecting a row loads its log. Rows are buttons rather than list items with a
 * click handler, so they are reachable and operable from the keyboard without
 * any ARIA at all.
 */
function JobRow({ record, watched, onWatch, onCancel }) {
  const tone = STATE_TONE[record.state] ?? ''
  return (
    <li>
      <button
        type="button"
        className={`job-row ${record.id === watched ? 'active' : ''}`}
        aria-pressed={record.id === watched}
        onClick={() => onWatch(record.id)}
      >
        <span className="job-name">{record.name}</span>
        <span className={`job-state ${tone}`}>{record.state}</span>
        <span className="job-time">{duration(record)}</span>
      </button>
      {record.state === 'running' && (
        <button type="button" className="ghost" onClick={() => onCancel(record.id)}>
          Stop
        </button>
      )}
    </li>
  )
}

/** Scrolls to the newest output only while the reader is already at the end. */
function LogView({ text }) {
  const ref = useRef(null)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    // Yanking someone back to the bottom while they are reading further up is
    // the reason log panels get called unusable.
    const atEnd = node.scrollHeight - node.scrollTop - node.clientHeight < 40
    if (atEnd) node.scrollTop = node.scrollHeight
  }, [text])

  return (
    <div
      className="scroll-body job-log"
      role="group"
      aria-label="Job output"
      tabIndex={0}
      ref={ref}
    >
      <pre>{text || 'No output yet. Some stages are quiet for minutes at a time.'}</pre>
    </div>
  )
}

/**
 * The pipeline stages, as buttons.
 *
 * Rendered only when `GET /capabilities` says this service exposes them, and
 * placed below diagnostics, outside `<main>`, for the same reason diagnostics
 * is: the two-column layout above is sized against the pitch, and a panel that
 * grew it would move the live surface every time a job printed a line.
 *
 * These stages take minutes and write to `data/` and `artifacts/`. The copy says
 * so rather than presenting them as ordinary buttons.
 */
export function PipelinePanel() {
  const { stages, jobs, running, pending, watched, log, start, cancel, watch } = useJobs(true)
  const [announcement, setAnnouncement] = useState('')

  const onStart = async (name) => {
    setAnnouncement(`Starting ${name}.`)
    const accepted = await start(name)
    setAnnouncement(accepted ? `${name} is running.` : `${name} could not be started.`)
  }

  const onCancel = async (id) => {
    await cancel(id)
    setAnnouncement('Stop requested.')
  }

  return (
    <section className="pipeline" id="pipeline" aria-labelledby="pipeline-heading">
      <h2 id="pipeline-heading">Pipeline</h2>
      <p className="panel-note">
        The same stages as the Makefile, run in a separate process. Each takes minutes and writes to{' '}
        <code>data/</code> and <code>artifacts/</code>. One at a time.
      </p>
      <p className="sr-only" role="status">
        {announcement}
      </p>

      <div className="stage-buttons">
        {stages.map((stage) => (
          <button
            key={stage.name}
            type="button"
            disabled={pending || Boolean(running)}
            aria-busy={running === stage.name}
            title={stage.description}
            onClick={() => onStart(stage.name)}
          >
            {stage.label}
          </button>
        ))}
      </div>

      <dl className="stage-notes">
        {stages.map((stage) => (
          <div key={stage.name}>
            <dt>{stage.label}</dt>
            <dd>
              {stage.description}
              {RELOADS[stage.name] ? (
                <span className="qualifier"> · on success, {RELOADS[stage.name]}</span>
              ) : null}
            </dd>
          </div>
        ))}
      </dl>

      <h3>Runs</h3>
      {jobs.length === 0 ? (
        <p className="empty">Nothing has been run yet.</p>
      ) : (
        <ul className="job-list">
          {jobs.map((record) => (
            <JobRow
              key={record.id}
              record={record}
              watched={watched}
              onWatch={watch}
              onCancel={onCancel}
            />
          ))}
        </ul>
      )}

      <LogView text={log} />
    </section>
  )
}
