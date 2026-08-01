import { useCallback, useEffect, useRef, useState } from 'react'
import Pitch from './Pitch.jsx'

const MAX_INSIGHTS = 5

function formatClock(seconds) {
  if (seconds === null || seconds === undefined) return '--:--'
  const total = Math.max(0, Math.floor(seconds))
  const mm = String(Math.floor(total / 60)).padStart(2, '0')
  const ss = String(total % 60).padStart(2, '0')
  return `${mm}:${ss}`
}

/**
 * Confidence bar with the decision threshold marked.
 *
 * The threshold is drawn because a bare probability is hard to read: what
 * matters to a viewer-facing product is whether the estimate has crossed the
 * line at which the system is willing to say something.
 */
function Confidence({ probability, threshold, isMl }) {
  const pct = probability === null ? 0 : Math.round(probability * 100)
  const unknown = probability === null
  return (
    <div className="confidence">
      <div className="confidence-head">
        <span>Model confidence</span>
        <span className="confidence-value">{unknown ? 'n/a' : `${pct}%`}</span>
      </div>
      <div className="meter">
        <div
          className={`meter-fill ${probability >= threshold ? 'over' : ''}`}
          style={{ width: `${pct}%` }}
        />
        <div className="meter-threshold" style={{ left: `${threshold * 100}%` }} />
      </div>
      {!isMl && (
        <div className="fallback-badge" title="Rule-based fallback, not a trained model">
          Fallback — not ML
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [frame, setFrame] = useState(null)
  const [probability, setProbability] = useState(null)
  const [insights, setInsights] = useState([])
  const [model, setModel] = useState(null)
  const [replay, setReplay] = useState(null)
  const [connected, setConnected] = useState(false)
  const [ready, setReady] = useState(null)
  const framesRef = useRef(0)

  useEffect(() => {
    fetch('/model').then((r) => (r.ok ? r.json() : null)).then(setModel).catch(() => {})
    fetch('/ready').then((r) => r.json()).then((d) => setReady(d.ready)).catch(() => setReady(false))
  }, [])

  useEffect(() => {
    const source = new EventSource('/insights/stream')
    source.onopen = () => setConnected(true)
    source.onerror = () => setConnected(false)
    source.addEventListener('update', (event) => {
      const message = JSON.parse(event.data)
      if (message.type === 'frame') {
        framesRef.current += 1
        setFrame(message.payload)
        setProbability(message.payload.window_valid ? message.payload.probability : null)
      } else if (message.type === 'insight') {
        setInsights((current) => [message.payload, ...current].slice(0, MAX_INSIGHTS))
      }
    })
    return () => source.close()
  }, [])

  useEffect(() => {
    const id = setInterval(() => {
      fetch('/replay/status').then((r) => (r.ok ? r.json() : null)).then(setReplay).catch(() => {})
    }, 2000)
    return () => clearInterval(id)
  }, [])

  const control = useCallback((body) => {
    fetch('/replay/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setReplay(d))
      .catch(() => {})
  }, [])

  const threshold = model?.decision_threshold ?? 0.5
  const isMl = model?.is_ml ?? false
  const dropped = replay?.fault_summary?.dropped ?? 0

  return (
    <div className="app">
      <header>
        <div>
          <h1>Live Football Insight Engine</h1>
          <p className="subtitle">
            Predicting imminent penalty-area entries from tracking data, and only saying so
            when it is worth saying.
          </p>
        </div>
        <div className={`status-pill ${ready ? 'ok' : 'bad'}`}>
          {ready === null ? 'checking' : ready ? 'ready' : 'not ready'}
        </div>
      </header>

      <main>
        <section className="pitch-panel">
          <Pitch frame={frame} attackingRight={null} />
          <div className="scrubber">
            <span className="clock">
              {formatClock(frame?.match_time_s)}
              <span className="period">{frame ? ` · P${frame.period}` : ''}</span>
            </span>
            <div className="controls">
              <button type="button" onClick={() => control({ paused: !replay?.paused })}>
                {replay?.paused ? 'Resume' : 'Pause'}
              </button>
              {[1, 2, 5, 10].map((s) => (
                <button
                  key={s}
                  type="button"
                  className={replay?.speed === s ? 'active' : ''}
                  onClick={() => control({ speed: s })}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>
        </section>

        <aside>
          <Confidence probability={probability} threshold={threshold} isMl={isMl} />

          <section className="insights">
            <h2>Insights</h2>
            {insights.length === 0 && (
              <p className="empty">
                No insight yet. The system stays quiet unless the estimate is strong enough
                and the situation is current.
              </p>
            )}
            {insights.map((insight, index) => (
              <article key={`${insight.match_time_s}-${index}`} className="insight">
                <div className="insight-time">{formatClock(insight.match_time_s)}</div>
                <div>
                  <div className="insight-headline">{insight.headline}</div>
                  {insight.detail && <div className="insight-detail">{insight.detail}</div>}
                  <div className="insight-meta">
                    {Math.round(insight.probability * 100)}% · {insight.attacking_team}
                    {!insight.is_ml && ' · fallback'}
                  </div>
                </div>
              </article>
            ))}
          </section>

          <section className="status">
            <h2>Status</h2>
            <dl>
              <div>
                <dt>Stream</dt>
                <dd className={connected ? 'ok' : 'bad'}>{connected ? 'connected' : 'offline'}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{model ? `${model.name} ${model.version}` : '—'}</dd>
              </div>
              <div>
                <dt>Feature schema</dt>
                <dd className={model?.schema_matches === false ? 'bad' : ''}>
                  {model?.feature_schema_hash?.slice(0, 8) ?? '—'}
                </dd>
              </div>
              <div>
                <dt>Fault profile</dt>
                <dd>
                  {replay ? `${replay.fault_profile} · seed ${replay.seed}` : '—'}
                </dd>
              </div>
              <div>
                <dt>Frames dropped</dt>
                <dd className={dropped > 0 ? 'warn' : ''}>{dropped}</dd>
              </div>
              <div>
                <dt>Replay</dt>
                <dd>
                  {replay ? `${replay.frames_emitted} / ${replay.total_frames}` : '—'}
                </dd>
              </div>
            </dl>
          </section>
        </aside>
      </main>

      <footer>
        Predictions are estimates over a short horizon, not statements of fact. Not affiliated
        with any league, broadcaster or data provider.
      </footer>
    </div>
  )
}
