import { useCallback, useEffect, useRef, useState } from 'react'

import { keyFor } from './format.js'

/** Matches what `GET /insights` returns, so a refresh restores the same depth. */
const MAX_INSIGHTS = 20
const REPLAY_POLL_MS = 2000
const SERVICE_POLL_MS = 5000

/**
 * Consecutive failures before a surface stops claiming to be current.
 *
 * One failed poll is noise — a dropped packet, a request that raced a restart.
 * Three in a row is a story, and the honest response is to keep showing the
 * last known values and say how old they are, not to blank a panel describing a
 * replay that is running perfectly well.
 */
const STALE_AFTER_FAILURES = 3

/** Frames a suppression reason must hold before it is worth displaying. */
const REASON_HOLD = 4

const DIAGNOSTICS_KEY = 'fi.diagnostics'

/**
 * Whether the diagnostics panel is open, remembered across reloads.
 *
 * Every access to storage is guarded: a browser can refuse it outright, and the
 * cost of that must be the preference, never the toggle.
 */
export function useDiagnosticsPreference() {
  const [open, setOpen] = useState(() => {
    try {
      return window.localStorage.getItem(DIAGNOSTICS_KEY) === '1'
    } catch {
      return false
    }
  })

  const toggle = useCallback(() => {
    setOpen((current) => {
      const next = !current
      try {
        window.localStorage.setItem(DIAGNOSTICS_KEY, next ? '1' : '0')
      } catch {
        // A refused write only costs the preference.
      }
      return next
    })
  }, [])

  return [open, toggle]
}

/**
 * Sparkline resolution.
 *
 * Frames arrive at 12.5 Hz; a 300-pixel sparkline cannot show that many
 * distinct points and re-rendering a long polyline twelve times a second is
 * work nobody can see. Every fourth frame is kept, carrying the *maximum* of
 * the four rather than the last: a spike that crossed the reporting line is the
 * one thing a trend must not hide.
 */
const SPARK_EVERY = 4
const SPARK_POINTS = 90

/**
 * Fetch JSON, returning the status alongside the body.
 *
 * The status is kept because a failed response is not always an absence:
 * `/ready` answers 503 with the reason it is not ready, and that reason is the
 * most useful thing on the page when the service is unhealthy. Only a transport
 * or parse failure gives `null` — every one of these calls is decoration around
 * a stream that is the real product, so the demo keeps drawing frames while
 * `/model` is unreachable.
 */
export async function fetchJson(url, options) {
  try {
    const response = await fetch(url, options)
    return {
      ok: response.ok,
      status: response.status,
      body: await response.json(),
    }
  } catch {
    return null
  }
}

/**
 * Newest-first insights, de-duplicated by identity and capped.
 *
 * Ordered by period before match time. In this dataset the clock runs straight
 * through the interval, so the two orderings agree — but a feed that silently
 * interleaved the halves if it ever did not would be a nasty thing to debug,
 * and stating the intended order costs nothing.
 */
function mergeInsights(existing, incoming) {
  const seen = new Set()
  const out = []
  for (const insight of [...incoming, ...existing]) {
    const key = keyFor(insight)
    if (seen.has(key)) continue
    seen.add(key)
    out.push(insight)
  }
  out.sort((a, b) => b.period - a.period || b.match_time_s - a.match_time_s)
  return out.slice(0, MAX_INSIGHTS)
}

/**
 * Model metadata and readiness.
 *
 * Both are fetched immediately and then polled. `/ready` polls for as long as
 * the page is open because it can flip: without that, a service that becomes
 * ready a second after load is described as broken until someone reloads.
 * `/model` stops once it succeeds — the metadata is fixed for the lifetime of
 * the process, so the retry exists only for a cold start racing the dev proxy.
 */
export function useServiceStatus() {
  const model = useModelMetadata()
  const { ready, readyReason } = useReadiness()
  return { model, ready, readyReason }
}

/** Retried until it answers, then left alone: it cannot change while we run. */
function useModelMetadata() {
  const [model, setModel] = useState(null)

  useEffect(() => {
    if (model !== null) return undefined
    let cancelled = false
    const poll = async () => {
      const response = await fetchJson('/model')
      if (!cancelled && response?.ok) setModel(response.body)
    }
    poll()
    const id = setInterval(poll, SERVICE_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [model])

  return model
}

/** Polled for as long as the page is open, because it can flip either way. */
function useReadiness() {
  const [ready, setReady] = useState(null)
  const [readyReason, setReadyReason] = useState(null)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      const response = await fetchJson('/ready')
      if (cancelled || response === null) return
      setReady(Boolean(response.body?.ready))
      setReadyReason(response.body?.reason ?? null)
    }
    poll()
    const id = setInterval(poll, SERVICE_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return { ready, readyReason }
}

/**
 * Subscribe to the server-sent event stream.
 *
 * The status is a five-value machine rather than a connected flag. A boolean
 * cannot tell a replay that reached its end from a service that died, and those
 * need opposite responses from the reader: one offers a restart, the other says
 * the backend is gone.
 *
 * A frame whose window was not structurally valid reports no probability at
 * all, rather than a stale one: nothing on screen may imply the model scored
 * something it declined to score.
 */
export function useInsightStream() {
  const [frame, setFrame] = useState(null)
  const [probability, setProbability] = useState(null)
  const [insights, setInsights] = useState([])
  const [status, setStatus] = useState('connecting')
  const [history, setHistory] = useState([])
  const [suppression, setSuppression] = useState({
    frames: 0,
    emitted: 0,
    counts: {},
  })
  const [reason, setReason] = useState(null)
  const [framesSeen, setFramesSeen] = useState(0)
  const [malformed, setMalformed] = useState(0)
  const [endedFrames, setEndedFrames] = useState(null)
  const [switchingTo, setSwitchingTo] = useState(null)
  const [generation, setGeneration] = useState(0)

  // Read inside event handlers, which close over the render that created them.
  const statusRef = useRef(status)
  statusRef.current = status
  const groupRef = useRef([])
  const errorsRef = useRef(0)

  const clearReplayState = useCallback(() => {
    setInsights([])
    setHistory([])
    setSuppression({ frames: 0, emitted: 0, counts: {} })
    setReason(null)
    setFramesSeen(0)
    setMalformed(0)
    setEndedFrames(null)
    groupRef.current = []
  }, [])

  /**
   * Reopen a closed stream.
   *
   * Bumping the generation re-runs the effect, which reconnects *and* re-seeds
   * from `GET /insights`. That second part is what makes this self-healing: a
   * client that already closed on `end` was not subscribed to hear the server's
   * `restart` message, so anything published between the rewind and the
   * resubscribe is recovered from history rather than lost.
   */
  const reopen = useCallback(() => {
    clearReplayState()
    setStatus('connecting')
    errorsRef.current = 0
    setGeneration((n) => n + 1)
  }, [clearReplayState])

  useEffect(() => {
    let cancelled = false

    fetchJson('/insights').then((response) => {
      if (cancelled || !response?.ok) return
      const seeded = response.body?.insights ?? []
      setInsights((current) => mergeInsights(current, seeded))
    })

    const source = new EventSource('/insights/stream')

    source.onopen = () => {
      errorsRef.current = 0
      setStatus('live')
    }

    source.onerror = () => {
      // A finished replay closes the connection on purpose. Treating that as a
      // fault is what made a clean end look like a crash.
      if (statusRef.current === 'finished') return
      errorsRef.current += 1
      setStatus(errorsRef.current >= STALE_AFTER_FAILURES ? 'offline' : 'reconnecting')
    }

    const onFrame = (payload) => {
      setFrame(payload)
      const scored = payload.window_valid ? payload.probability : null
      setProbability(scored)
      setReason(payload.suppression ?? null)
      setFramesSeen((n) => n + 1)

      const group = groupRef.current
      group.push(scored)
      if (group.length < SPARK_EVERY) return
      const values = group.filter((p) => p !== null && p !== undefined)
      const point = values.length ? Math.max(...values) : null
      groupRef.current = []
      setHistory((current) => [...current, point].slice(-SPARK_POINTS))
    }

    const onSuppression = (payload) => {
      setSuppression((current) => {
        const counts = { ...current.counts }
        for (const [key, value] of Object.entries(payload.counts ?? {})) {
          counts[key] = (counts[key] ?? 0) + value
        }
        return {
          frames: current.frames + (payload.frames ?? 0),
          emitted: current.emitted + (payload.emitted ?? 0),
          counts,
        }
      })
    }

    // One handler per message type, looked up rather than tested in sequence:
    // the server's discriminator has six values now, and a chain of branches is
    // both harder to read and easy to extend in the wrong place.
    const handlers = {
      frame: onFrame,
      insight: (payload) => setInsights((current) => mergeInsights(current, [payload])),
      suppression: onSuppression,
      // The frame is deliberately kept until the next one arrives, so the pitch
      // does not blank for a moment on every restart.
      restart: () => {
        clearReplayState()
        setStatus('live')
      },
      // Sent by the server, not inferred from the click, so every open tab
      // follows the change rather than only the one that asked for it. The
      // stream stays open throughout: the connection is fine, the match behind
      // it is being replaced.
      match: (payload) => {
        clearReplayState()
        setSwitchingTo(payload?.loading ? (payload?.match_id ?? null) : null)
        setStatus('live')
      },
      // Closing is mandatory. The server returns from its publisher on the end
      // marker and closes the response; an EventSource left open reconnects a
      // few seconds later, the service starts a fresh replay task whose player
      // is already at the last frame, and it ends again — forever.
      end: (payload) => {
        source.close()
        setStatus('finished')
        setEndedFrames(payload?.frames ?? null)
      },
    }

    source.addEventListener('update', (event) => {
      let message
      try {
        message = JSON.parse(event.data)
      } catch {
        // Counted rather than swallowed. A malformed payload is a transport
        // defect, and a demo that quietly drops them is why nobody finds it.
        setMalformed((n) => n + 1)
        return
      }
      handlers[message.type]?.(message.payload)
    })

    return () => {
      cancelled = true
      source.close()
    }
  }, [generation, clearReplayState])

  return {
    frame,
    probability,
    insights,
    status,
    history,
    suppression,
    reason,
    framesSeen,
    malformed,
    endedFrames,
    switchingTo,
    reopen,
  }
}

/**
 * Hold a suppression reason until a new one has persisted.
 *
 * The reason is decided per frame, so the raw field flickers between two or
 * three values faster than any of them can be read. A reason is adopted only
 * once it has been the answer four frames running — about a third of a second
 * of match time, which is roughly the shortest label a reader can take in.
 *
 * `tick` is the frame counter rather than the reason itself: consecutive frames
 * usually carry the *same* reason, so an effect keyed on the value alone would
 * never re-run and the run length would never grow.
 */
export function useStickyReason(reason, tick) {
  const [sticky, setSticky] = useState(null)
  const runRef = useRef({ value: undefined, count: 0, tick: -1 })

  useEffect(() => {
    const run = runRef.current
    if (tick === run.tick) return
    // The counter restarting means the replay did; nothing carries over.
    if (tick < run.tick) {
      runRef.current = { value: undefined, count: 0, tick }
      setSticky(null)
      return
    }
    run.tick = tick
    if (reason === run.value) {
      run.count += 1
    } else {
      run.value = reason
      run.count = 1
    }
    if (run.count >= REASON_HOLD) setSticky(reason ?? null)
  }, [reason, tick])

  return sticky
}

/**
 * Poll replay position and fault summary, and expose pause, speed and restart.
 *
 * A failed poll never overwrites a good reading. Blanking the panel on one
 * dropped request describes the replay as unknown when it is running fine; after
 * three failures the panel says how stale it is instead, which is both honest
 * and more useful.
 */
function useReplayStatus() {
  const [replay, setReplay] = useState(null)
  const [stale, setStale] = useState(false)
  const [lastSeen, setLastSeen] = useState(null)
  const failuresRef = useRef(0)

  const accept = useCallback((body) => {
    failuresRef.current = 0
    setReplay(body)
    setStale(false)
    setLastSeen(Date.now())
  }, [])

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      const response = await fetchJson('/replay/status')
      if (cancelled) return
      if (response?.ok) {
        accept(response.body)
        return
      }
      failuresRef.current += 1
      if (failuresRef.current >= STALE_AFTER_FAILURES) setStale(true)
    }

    poll()
    const id = setInterval(poll, REPLAY_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [accept])

  return { replay, stale, lastSeen, accept }
}

/** The match catalogue, fixed for the lifetime of the process, so read once. */
function useMatchCatalogue() {
  const [matches, setMatches] = useState([])

  useEffect(() => {
    let cancelled = false
    fetchJson('/replay/matches').then((response) => {
      if (!cancelled && response?.ok) setMatches(response.body?.matches ?? [])
    })
    return () => {
      cancelled = true
    }
  }, [])

  return matches
}

export function useReplayControl() {
  const { replay, stale, lastSeen, accept } = useReplayStatus()
  const matches = useMatchCatalogue()
  const [pending, setPending] = useState(false)

  /**
   * Send one command, blocking further commands until it resolves.
   *
   * The guard is not cosmetic: two restarts fired by an impatient double-click
   * would rewind the replay twice, and the second rewind lands while the loop is
   * still applying the first. Changing match runs through the same guard and
   * needs it more, not less — that request holds for about two seconds while the
   * new match is parsed.
   */
  const send = useCallback(
    async (url, body) => {
      setPending(true)
      try {
        const response = await fetchJson(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (response?.ok) accept(response.body)
        return Boolean(response?.ok)
      } finally {
        setPending(false)
      }
    },
    [accept],
  )

  const control = useCallback((body) => send('/replay/control', body), [send])
  const restart = useCallback(() => send('/replay/control', { restart: true }), [send])
  const selectMatch = useCallback((match) => send('/replay/match', { match }), [send])

  return {
    replay,
    matches,
    control,
    restart,
    selectMatch,
    pending,
    stale,
    lastSeen,
  }
}

/**
 * Restart and change-match, each reopening the stream only when it must.
 *
 * A live client hears the server's `restart` and `match` messages and needs no
 * reconnect. A *finished* one closed its stream on the end marker, so it is no
 * longer subscribed and would sit on the old replay forever; reopening also
 * re-seeds from `GET /insights`, recovering anything published in between.
 */
export function useReplayActions({ restart, selectMatch, reopen, status }) {
  const onRestart = useCallback(async () => {
    const accepted = await restart()
    if (accepted) reopen()
    return accepted
  }, [restart, reopen])

  const onSelectMatch = useCallback(
    async (id) => {
      const accepted = await selectMatch(id)
      if (accepted && status === 'finished') reopen()
      return accepted
    },
    [selectMatch, reopen, status],
  )

  return { onRestart, onSelectMatch }
}
