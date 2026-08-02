/**
 * Structural assertions for the viewer surface, rendered on the server.
 *
 * These prove *structure*, never geometry: that the live region survived being
 * wrapped in a scrolling body, that the confidence panel offers the same slots
 * in every state, and that no percentage reaches an audience. Whether the
 * layout actually holds still is a question about a rendered page and is
 * answered by measuring one in a browser, not here.
 *
 * Built with vite (`npm run test:ssr`) because the sources are JSX; run with
 * `node --test`. Nothing is installed for this — `react-dom/server` ships with
 * the react-dom the demo already depends on.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup } from 'react-dom/server'

import App from '../src/App.jsx'
import { Confidence, Header, RuntimeStatus } from '../src/panels.jsx'
import { Transport } from '../src/transport.jsx'
import { InsightList } from '../src/insights.jsx'
import { PipelinePanel } from '../src/pipeline.jsx'
import { keyFor } from '../src/format.js'

const FACTS = [
  { key: 'attackers_ahead', text: '3 attackers ahead of the ball', value: 3 },
  { key: 'nearest_defender', text: 'nearest defender 8 m away', value: 8 },
]

/** `n` insights shaped like the service's, newest first. */
function insights(n) {
  return Array.from({ length: n }, (_, i) => ({
    kind: ['building_threat', 'elevated_entry_chance', 'sustained_pressure'][i % 3],
    headline: 'Attacking threat is building',
    detail: 'Estimated over the next 8 seconds.',
    probability: 0.7,
    match_time_s: 1200 - i * 37.5,
    period: 1,
    attacking_team: i % 2 ? 'home' : 'away',
    is_ml: true,
    facts: i % 2 ? FACTS : [],
  }))
}

const render = (element) => renderToStaticMarkup(element)

/** Markup with every tag removed, which is what a reader is actually shown. */
const visibleText = (markup) => markup.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ')

/** The part of the panel that scrolls, and the fixed shell around it. */
function split(markup) {
  const open = markup.indexOf('<div class="scroll-body"')
  assert.notEqual(open, -1, 'the insight panel has no scroll body')
  return { shell: markup.slice(0, open), body: markup.slice(open) }
}

const list = (props) =>
  render(
    <InsightList
      insights={[]}
      status="live"
      endedFrames={null}
      reason={null}
      framesSeen={0}
      {...props}
    />,
  )

test('the live region keeps its semantics inside the scroll body', () => {
  const markup = list({ insights: insights(3) })
  const { body } = split(markup)

  assert.match(body, /<ol aria-live="polite" aria-relevant="additions" aria-atomic="false">/)
  assert.equal(markup.match(/<ol\b/g).length, 1, 'exactly one list')
})

test('the scroll body is a named region a keyboard can reach', () => {
  const markup = list({ insights: insights(3) })

  assert.match(markup, /<div class="scroll-body" role="group" aria-label="[^"]+" tabindex="0">/)
})

test('the heading and the count stay outside the scrolling part', () => {
  const { shell, body } = split(list({ insights: insights(5) }))

  assert.match(shell, /<h2 id="insights-heading">Insights<\/h2>/)
  assert.match(shell, /class="insight-count">5 insights, newest first</)
  assert.doesNotMatch(body, /insight-count/)
  assert.doesNotMatch(body, /<h2/)
})

test('the count is not inside a live region, so it is not announced per frame', () => {
  const markup = list({ insights: insights(5) })
  const before = markup.slice(0, markup.indexOf('insight-count'))

  assert.doesNotMatch(before, /aria-live/)
})

test('an empty feed states why, inside the body, with no items', () => {
  const markup = list({ insights: [], reason: 'cooldown', framesSeen: 40 })
  const { body } = split(markup)

  assert.match(body, /class="empty"/)
  assert.match(body, /too soon after the last one/)
  assert.equal(markup.match(/class="insight"/g), null)
})

test('a finished replay is reported in the fixed shell, not in the history', () => {
  const { shell, body } = split(list({ insights: insights(20), status: 'finished', endedFrames: 2790 }))

  assert.match(shell, /class="finished"/)
  assert.match(shell, /Replay finished after 2,790 frames/)
  assert.doesNotMatch(body, /class="finished"/)
})

test('twenty insights render twenty items with distinct keys', () => {
  const twenty = insights(20)
  const markup = list({ insights: twenty })

  assert.equal(markup.match(/<li class="insight">/g).length, 20)
  assert.equal(new Set(twenty.map(keyFor)).size, 20)
})

const confidence = (props) =>
  render(
    <Confidence
      probability={0.2}
      threshold={0.5}
      history={[]}
      isMl
      reason={null}
      horizonS={8}
      {...props}
    />,
  )

/** The structural slots, in order, ignoring whatever copy is in them. */
const slots = (markup) =>
  (markup.match(/class="[^"]*"/g) ?? []).filter((c) => !c.includes('meter-fill'))

test('the confidence panel offers the same slots in every scoring state', () => {
  const unscored = slots(confidence({ probability: null }))
  const quiet = slots(confidence({ probability: 0.2 }))
  const reporting = slots(confidence({ probability: 0.9 }))

  // Band classes differ by design; the slots that hold copy must not.
  const structural = (list) => list.filter((c) => !c.startsWith('class="band band-'))
  assert.deepEqual(structural(quiet), structural(unscored))
  assert.deepEqual(structural(reporting), structural(unscored))
})

test('the reserved slots exist whether or not there is anything to put in them', () => {
  for (const markup of [confidence({ reason: null }), confidence({ reason: 'cooldown' })]) {
    assert.match(markup, /class="panel-note confidence-lede"/)
    assert.match(markup, /class="panel-note confidence-detail"/)
    assert.match(markup, /class="panel-note confidence-trend"/)
    assert.match(markup, /class="confidence-foot"/)
  }
})

test('the fallback badge waits until the service has actually answered', () => {
  assert.doesNotMatch(confidence({ isMl: null }), /fallback-badge/)
  assert.doesNotMatch(confidence({ isMl: true }), /fallback-badge/)
  assert.match(confidence({ isMl: false }), /fallback-badge/)
})

test('diagnostics stay below main rather than beside or inside it', () => {
  // The stored preference is the only way the panel is open on first render.
  globalThis.window = { localStorage: { getItem: () => '1', setItem: () => {} } }
  try {
    const markup = render(<App />)
    const closed = markup.indexOf('</main>')

    assert.ok(closed > 0, 'no main element')
    assert.ok(
      markup.indexOf('id="diagnostics"') > closed,
      'diagnostics must follow main, not sit inside it',
    )
    assert.match(markup.slice(0, closed), /class="pitch-panel"/)
    assert.match(markup.slice(0, closed), /id="insights"/)
  } finally {
    delete globalThis.window
  }
})

const MATCHES = [
  { id: 'Sample_Game_1', source_format: 'metrica_csv', available: true },
  { id: 'Sample_Game_2', source_format: 'metrica_csv', available: true },
  { id: 'Sample_Game_3', source_format: 'metrica_epts', available: false },
]

const header = (props) =>
  render(
    <Header
      ready
      readyReason="ok"
      showDiagnostics={false}
      onToggleDiagnostics={() => {}}
      matches={MATCHES}
      currentMatch="Sample_Game_2"
      onSelectMatch={() => {}}
      switching={false}
      {...props}
    />,
  )

test('the match picker is labelled and shows the match being replayed', () => {
  const markup = header()

  assert.match(markup, /<label for="match-select">Match<\/label>/)
  assert.match(markup, /<option value="Sample_Game_2" selected="">Sample Game 2<\/option>/)
})

test('a match that has not been downloaded is offered but not selectable', () => {
  const markup = header()

  assert.match(markup, /<option value="Sample_Game_3" disabled="">Sample Game 3 — not downloaded/)
})

test('the picker is absent until the catalogue arrives, and disabled while switching', () => {
  assert.doesNotMatch(header({ matches: [] }), /match-select/)
  assert.match(header({ switching: true }), /<select id="match-select"[^>]*disabled=""/)
})

test('a match being loaded is named where the direction normally goes', () => {
  const transport = (props) =>
    render(
      <Transport
        frame={null}
        replay={null}
        control={() => {}}
        restart={() => {}}
        pending={false}
        status="live"
        stale={false}
        lastSeen={null}
        {...props}
      />,
    )

  // The direction slot is always present, so using it costs no height — the
  // pitch beside it must not move while a match loads.
  assert.match(transport({ switchingTo: 'Sample_Game_1' }), /Loading Sample Game 1/)
  assert.match(transport({}), /Direction unknown/)
  const idle = transport({}).match(/class="[^"]*"/g)
  const loading = transport({ switchingTo: 'Sample_Game_1' }).match(/class="[^"]*"/g)
  assert.deepEqual(loading, idle, 'the loading state must add no elements')
})

test('the pipeline panel renders below main, never inside it', () => {
  const markup = render(<PipelinePanel />)

  assert.match(markup, /<section class="pipeline" id="pipeline"/)
  assert.doesNotMatch(markup, /<main/)
  // Contained the same way the insight feed is, so a long log cannot grow the page.
  assert.match(markup, /class="scroll-body job-log"/)
})

test('the pipeline panel says what these stages cost before anyone presses one', () => {
  const text = visibleText(render(<PipelinePanel />))

  assert.match(text, /takes minutes/)
  assert.match(text, /data\/ and artifacts\//)
})

const SYNTHETIC_RUNTIME = {
  mode: 'public_demo',
  dataSource: 'synthetic',
  predictor: { name: 'demo-synthetic-gru', kind: 'gru', is_ml: true },
  replay: 'running',
}

test('the status block renders nothing until the service has answered', () => {
  // Rendering placeholders would put "Unknown" under every label on each load,
  // and a viewer reading the page at that moment would be told the wrong thing
  // about the data rather than nothing at all.
  assert.equal(render(<RuntimeStatus runtime={null} />), '')
  assert.equal(render(<RuntimeStatus runtime={undefined} />), '')
})

test('the status block says generated data is not match data', () => {
  const text = visibleText(render(<RuntimeStatus runtime={SYNTHETIC_RUNTIME} />))

  assert.match(text, /Public demo/)
  assert.match(text, /Generated fixture \(not real match data\)/)
  assert.match(text, /Sequence model \(GRU\) — demo-synthetic-gru/)
  // The one label that must never appear against a synthetic fixture.
  assert.doesNotMatch(text, /Metrica/)
})

test('the status block names the fallback as a fallback', () => {
  const runtime = {
    ...SYNTHETIC_RUNTIME,
    predictor: { name: 'heuristic-fallback', kind: 'heuristic', is_ml: false },
  }
  const markup = render(<RuntimeStatus runtime={runtime} />)

  assert.match(visibleText(markup), /Rule-based fallback/)
  // Flagged, not merely stated: this is the case a reader must not skim past.
  assert.match(markup, /class="warn"[^>]*>Rule-based fallback/)
})

test('a real match is labelled as one', () => {
  const runtime = {
    mode: 'local',
    dataSource: 'metrica',
    predictor: { name: 'gru-temporal', kind: 'gru', is_ml: true },
    replay: 'running',
  }
  const text = visibleText(render(<RuntimeStatus runtime={runtime} />))

  assert.match(text, /Metrica sample match/)
  assert.doesNotMatch(text, /Generated fixture/)
})

test('the footer states what this is not', () => {
  const text = visibleText(render(<App />))

  assert.match(text, /not an injury, betting, officiating or player-safety system/)
  // The evaluated model and the hosted one are different artifacts, and the
  // page has to say so where the numbers might otherwise be assumed to apply.
  assert.match(text, /measured on Metrica sample matches and do not describe the model serving/)
})

test('no percentage is shown to an audience', () => {
  const surfaces = [
    confidence({ probability: 0.87 }),
    confidence({ probability: null }),
    list({ insights: insights(5) }),
  ]

  for (const markup of surfaces) {
    assert.doesNotMatch(visibleText(markup), /\d\s*%/)
  }
})
