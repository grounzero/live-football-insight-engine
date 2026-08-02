import { useCallback, useEffect, useState } from 'react'

import { fetchJson } from './hooks.js'

/** Polled fast while something is running, slowly the rest of the time. */
const POLL_ACTIVE_MS = 1500
const POLL_IDLE_MS = 10000

/**
 * Characters of job output kept in memory.
 *
 * A training run can emit far more than a page needs, and an unbounded string
 * grows the document the same way the insight feed used to. The tail is the
 * useful end.
 */
const LOG_CHARS = 20000

/**
 * Which optional surfaces this deployment exposes.
 *
 * Asked once and never polled: it is decided when the process starts. `null`
 * until the answer arrives, so nothing renders on the assumption a capability
 * is present — the pipeline panel would otherwise flash into view on every load
 * of a service that does not have it.
 */
export function useCapabilities() {
  const [capabilities, setCapabilities] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchJson('/capabilities').then((response) => {
      if (!cancelled && response?.ok) setCapabilities(response.body)
    })
    return () => {
      cancelled = true
    }
  }, [])

  return capabilities
}

/** The stages this build offers and the runs recorded so far. */
function useJobListing(enabled) {
  const [listing, setListing] = useState({
    stages: [],
    jobs: [],
    running: null,
  })

  const refresh = useCallback(async () => {
    const response = await fetchJson('/jobs')
    if (!response?.ok) return null
    const body = response.body ?? {}
    setListing({
      stages: body.stages ?? [],
      jobs: body.jobs ?? [],
      running: body.running ?? null,
    })
    return body.running ?? null
  }, [])

  useEffect(() => {
    if (!enabled) return undefined
    let cancelled = false
    let timer = null
    // Chained timeouts rather than an interval: the cadence depends on the
    // answer, and an interval would have to be torn down and rebuilt to change.
    const tick = async () => {
      const running = await refresh()
      if (cancelled) return
      timer = setTimeout(tick, running ? POLL_ACTIVE_MS : POLL_IDLE_MS)
    }
    tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [enabled, refresh])

  return { ...listing, refresh }
}

/** Tail one job's output, bounded, closing when the server says it is over. */
function useJobLog(watched, onDone) {
  const [log, setLog] = useState('')

  useEffect(() => {
    if (!watched) return undefined
    setLog('')
    const source = new EventSource(`/jobs/${encodeURIComponent(watched)}/log`)

    const onMessage = (event) => {
      let message
      try {
        message = JSON.parse(event.data)
      } catch {
        return
      }
      if (message.type === 'log') {
        setLog((current) => (current + message.text).slice(-LOG_CHARS))
        return
      }
      // The server closes after `done`; closing here too stops the browser
      // reconnecting to a finished job for the life of the page.
      source.close()
      onDone()
    }

    source.addEventListener('update', onMessage)
    source.onerror = () => source.close()
    return () => source.close()
  }, [watched, onDone])

  return log
}

/**
 * The pipeline stages, the runs so far, and the output of one of them.
 *
 * Inert unless `enabled`, which comes from `/capabilities`: on a service without
 * the pipeline surface these endpoints do not exist, and polling them would be
 * a 404 every few seconds for as long as the page is open.
 */
export function useJobs(enabled) {
  const { stages, jobs, running, refresh } = useJobListing(enabled)
  const [pending, setPending] = useState(false)
  const [watched, setWatched] = useState(null)
  const log = useJobLog(enabled ? watched : null, refresh)

  // Follow whatever is running; a reader can still select a finished run.
  useEffect(() => {
    if (running) setWatched(running)
  }, [running])

  const post = useCallback(
    async (path, follow) => {
      setPending(true)
      try {
        const response = await fetchJson(path, { method: 'POST' })
        if (!response?.ok) return false
        if (follow) setWatched(response.body?.id ?? null)
        await refresh()
        return true
      } finally {
        setPending(false)
      }
    },
    [refresh],
  )

  const start = useCallback((name) => post(`/jobs/${encodeURIComponent(name)}`, true), [post])
  const cancel = useCallback((id) => post(`/jobs/${encodeURIComponent(id)}/cancel`, false), [post])

  return {
    stages,
    jobs,
    running,
    pending,
    watched,
    log,
    start,
    cancel,
    watch: setWatched,
  }
}
