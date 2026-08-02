/**
 * Playback: the clock, how far through the replay is, and the controls.
 *
 * Its own module rather than another entry in `panels.jsx`, which had become a
 * drawer: the viewer-facing panels there describe what the model thinks, and
 * this describes what the replay is doing.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { directionText } from './Pitch.jsx'
import { ABSENT, formatClock, formatCount } from './format.js'

const SPEEDS = [1, 2, 5, 10]

/**
 * What stands in for the controls on a public deployment.
 *
 * The rate is stated because the replay is faster than match pace and a viewer
 * who assumed otherwise would be judging the football rather than the pipeline
 * — every player would look impossibly quick. It comes from the running
 * service rather than being written here, so the sentence cannot outlive a
 * change to the deployment, and it is omitted entirely until the service has
 * answered rather than guessed at.
 *
 * The reason the controls are absent is given, not just their absence: replay
 * state is process-wide, so one visitor pausing would pause it for everyone.
 */
export function ReadOnlyNotice({ speed = null }) {
  return (
    <p className="transport-readonly">
      Server-controlled public replay
      {speed ? ` at ${speed}× match pace` : ''}, looping continuously. Playback controls are
      disabled because every viewer shares one replay.
    </p>
  )
}

/** How long a spoken confirmation stays in the live region, in milliseconds. */
const ANNOUNCE_MS = 4000

/**
 * A short spoken confirmation that clears itself.
 *
 * Screen-reader users get no feedback from a button whose effect is a change in
 * a canvas, so each command says what it did. It is cleared afterwards so a
 * stale sentence is not re-read on the next unrelated update.
 */
function useAnnouncement() {
  const [announcement, setAnnouncement] = useState('')
  const timerRef = useRef(null)

  const announce = useCallback((message) => {
    setAnnouncement(message)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setAnnouncement(''), ANNOUNCE_MS)
  }, [])

  useEffect(() => () => timerRef.current && clearTimeout(timerRef.current), [])

  return [announcement, announce]
}

/** Each command, paired with the sentence it says once the service accepts it. */
function useTransportCommands({ control, restart, paused, announce }) {
  const togglePause = useCallback(async () => {
    const next = !paused
    if (await control({ paused: next })) announce(next ? 'Replay paused.' : 'Replay resumed.')
  }, [control, paused, announce])

  const onRestart = useCallback(async () => {
    if (await restart()) announce('Replay restarted.')
  }, [restart, announce])

  const onSpeed = useCallback(
    async (speed) => {
      if (await control({ speed })) announce(`Speed set to ${speed} times real time.`)
    },
    [control, announce],
  )

  const onKeyDown = useCallback(
    (event) => {
      if (event.key !== ' ' && event.key !== 'Spacebar') return
      // A focused button already handles Space itself; intercepting it here
      // would fire two actions from one press.
      if (event.target.closest('button')) return
      event.preventDefault()
      togglePause()
    },
    [togglePause],
  )

  return { togglePause, onRestart, onSpeed, onKeyDown }
}

/** Replay position, with every absent reading resolved to a drawable default. */
function position(replay) {
  const emitted = replay?.frames_emitted ?? 0
  const total = replay?.total_frames ?? 0
  return {
    paused: replay?.paused ?? false,
    emitted,
    total,
    progress: total > 0 ? Math.min(100, Math.round((emitted / total) * 100)) : 0,
  }
}

function staleNote(lastSeen) {
  if (!lastSeen) return 'no reading yet'
  return `last seen ${Math.round((Date.now() - lastSeen) / 1000)}s ago`
}

/**
 * The buttons, and the live speed the buttons cannot express.
 *
 * `busy` is one flag rather than each button repeating the same three
 * conditions: while a match is loading there is no player to command, and a
 * control that still looked pressable would silently do nothing.
 */
function TransportControls({
  paused,
  finished,
  busy,
  pending,
  speed,
  stale,
  lastSeen,
  onTogglePause,
  onRestart,
  onSpeed,
}) {
  return (
    <div className="controls">
      <button type="button" onClick={onTogglePause} disabled={busy || finished} aria-busy={pending}>
        {paused ? 'Resume' : 'Pause'}
      </button>
      <button
        type="button"
        className={finished ? 'primary' : ''}
        onClick={onRestart}
        disabled={busy}
        aria-busy={pending}
      >
        Restart
      </button>

      <div className="speeds" role="group" aria-label="Replay speed">
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            className={speed === s ? 'active' : ''}
            aria-pressed={speed === s}
            aria-label={`${s} times real time`}
            disabled={busy}
            onClick={() => onSpeed(s)}
          >
            {s}x
          </button>
        ))}
      </div>

      {/* The service may be running at a speed no button offers — `make serve`
          uses 8x — so the live value is stated rather than left unexplained. */}
      <span className="speed-live">
        Speed {speed === undefined ? ABSENT : `${speed}×`}
        {stale ? ` · stale, ${staleNote(lastSeen)}` : ''}
      </span>
    </div>
  )
}

/**
 * Clock, progress and playback controls.
 *
 * Named for what it does. It was called a scrubber, but there is no seek behind
 * it and there cannot be one without a position setter on the player, so the
 * progress bar is explicitly non-interactive: no handle, no pointer cursor, and
 * `role="progressbar"` rather than `role="slider"`. A control that looks
 * draggable is a promise the service cannot keep.
 */
export function Transport({
  frame,
  replay,
  control,
  restart,
  pending,
  status,
  stale,
  lastSeen,
  switchingTo,
}) {
  const [announcement, announce] = useAnnouncement()
  const { paused, emitted, total, progress } = position(replay)
  const finished = status === 'finished'
  const busy = pending || Boolean(switchingTo)
  // The direction slot is always present and, while a match is loading, its
  // answer is genuinely unknown — so the loading notice goes there rather than
  // into a line that appears and disappears and moves the pitch with it.
  const direction = switchingTo
    ? `Loading ${switchingTo.replace(/_/g, ' ')}…`
    : directionText(frame)

  const { togglePause, onRestart, onSpeed, onKeyDown } = useTransportCommands({
    control,
    restart,
    paused,
    announce,
  })

  return (
    <div
      className="transport"
      role="group"
      aria-label="Replay transport"
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      <p className="sr-only" role="status">
        {announcement}
      </p>

      <div className="transport-head">
        <span className="clock">
          {formatClock(frame?.match_time_s)}
          <span className="period">{frame ? ` · P${frame.period}` : ''}</span>
        </span>
        <span className="direction">{direction ?? 'Direction unknown'}</span>
      </div>

      <div
        className="progress"
        role="progressbar"
        aria-label="Replay position"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={progress}
        aria-valuetext={`${formatCount(emitted)} of ${formatCount(total)} frames`}
      >
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <TransportControls
        paused={paused}
        finished={finished}
        busy={busy}
        pending={pending}
        speed={replay?.speed}
        stale={stale}
        lastSeen={lastSeen}
        onTogglePause={togglePause}
        onRestart={onRestart}
        onSpeed={onSpeed}
      />
    </div>
  )
}
