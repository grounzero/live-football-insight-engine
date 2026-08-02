import { formatClock, formatCount, keyFor, suppressionLabel } from './format.js'

/**
 * The measured observations behind an insight.
 *
 * The service sends these as structured `{key, text, value}` records and the
 * demo used to throw them away in favour of the pre-joined sentence. Shown as
 * separate chips each one can be checked against the pitch beside it, which is
 * the whole argument for keeping facts apart from the hedged headline: the
 * headline is an estimate, these are things that were actually measured.
 */
function FactChips({ facts }) {
  if (!facts?.length) return null
  return (
    <ul className="facts">
      {facts.map((fact) => (
        <li key={fact.key} className="fact">
          {fact.text}
        </li>
      ))}
    </ul>
  )
}

function InsightCard({ insight }) {
  return (
    <li className="insight">
      <div className="insight-time">{formatClock(insight.match_time_s)}</div>
      <div>
        <div className="insight-headline">{insight.headline}</div>
        <FactChips facts={insight.facts} />
        <div className="insight-meta">
          {insight.attacking_team}
          {!insight.is_ml && ' · fallback'}
        </div>
      </div>
    </li>
  )
}

/** Shown once the replay reaches its end, so a clean finish never reads as a fault. */
function FinishedCard({ frames }) {
  return (
    <p className="finished">
      Replay finished{frames ? ` after ${formatCount(frames)} frames` : ''}. Use Restart to play it
      again.
    </p>
  )
}

/** Static ordering label and depth, stated once rather than announced. */
function countLabel(count) {
  if (count === 0) return 'Newest first'
  return `${formatCount(count)} ${count === 1 ? 'insight' : 'insights'}, newest first`
}

/**
 * The insight feed.
 *
 * A polite live region announcing additions only. Prepending into an
 * `additions` region is exactly what it exists for: a new insight is read once,
 * and the rest of the list is not read again with it. This is the one place a
 * reader who cannot see the pitch gets the system's actual output.
 *
 * The panel is a fixed shell around a scrolling body. Twenty insights measured
 * 2,092 px of list, which grew the page to 2,567 px and put the diagnostics
 * panel 2,525 px down — a history nobody asked for, burying the engineering
 * surface under it. Only the list scrolls: the heading, the count and the
 * finished notice stay put, so a reader who has scrolled back through the
 * history can still see that the replay has ended.
 *
 * The count is plain text outside the live region on purpose. Inside it, every
 * arriving insight would re-announce the tally as well as the insight.
 */
export function InsightList({ insights, status, endedFrames, reason, framesSeen }) {
  const quiet = insights.length === 0

  return (
    <section className="insights" id="insights" aria-labelledby="insights-heading">
      <h2 id="insights-heading">Insights</h2>

      {status === 'finished' && <FinishedCard frames={endedFrames} />}

      <p className="insight-count">{countLabel(insights.length)}</p>

      {/* `tabIndex` is not decoration: an insight card holds nothing focusable,
          so without a tab stop this region cannot be scrolled from the keyboard
          in WebKit. Chrome and Firefox make scrollers focusable on their own;
          Safari does not, and a history you cannot reach is not a history. */}
      <div className="scroll-body" role="group" aria-label="Insight history" tabIndex={0}>
        {quiet && (
          <p className="empty">
            No insight yet. The system stays quiet unless the estimate is strong enough and the
            situation is current.
            {framesSeen > 0 && reason && <> Right now: {suppressionLabel(reason).toLowerCase()}.</>}
          </p>
        )}

        <ol aria-live="polite" aria-relevant="additions" aria-atomic="false">
          {insights.map((insight) => (
            <InsightCard key={keyFor(insight)} insight={insight} />
          ))}
        </ol>
      </div>
    </section>
  )
}
