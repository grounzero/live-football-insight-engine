/**
 * Pure formatting and vocabulary shared by the viewer surface and the
 * diagnostics panel.
 *
 * Kept out of the JSX modules because the diagnostics panel needs the same
 * clock, the same absent-value marker and the same suppression vocabulary as
 * the main view. Importing those from a sibling component module would couple
 * two unrelated surfaces through a concern that is neither's.
 */

/** Shown wherever a value is genuinely not known yet. */
export const ABSENT = '—'

/** Format seconds from kick-off as mm:ss. */
export function formatClock(seconds) {
  if (seconds === null || seconds === undefined) return '--:--'
  const total = Math.max(0, Math.floor(seconds))
  const mm = String(Math.floor(total / 60)).padStart(2, '0')
  const ss = String(total % 60).padStart(2, '0')
  return `${mm}:${ss}`
}

/** Group thousands, so six-figure frame counts stay readable. */
export function formatCount(value) {
  return typeof value === 'number' ? value.toLocaleString('en-GB') : ABSENT
}

/** Render a value only when its source is present, otherwise show a dash. */
export function describe(source, format) {
  return source ? format(source) : ABSENT
}

/**
 * The state the estimate is in, as the system itself understands it.
 *
 * The decision threshold is the only boundary the service actually acts on, so
 * it is the only boundary shown. `null` is not "zero": a window the model
 * declined to score has no estimate at all, and a bar sitting at the far left
 * would be claiming one.
 */
export function bandFor(probability, threshold) {
  if (probability === null || probability === undefined) return 'unscored'
  return probability >= threshold ? 'reporting' : 'quiet'
}

/**
 * Viewer-facing copy for each band.
 *
 * Deliberately no percentages. `docs/model_card.md` is explicit that the raw
 * probability must not be surfaced to an audience as one: the model is trained
 * with a ~16:1 positive class weight, so its output ranks well but is not a
 * frequency. What a viewer can honestly be told is which side of the reporting
 * line the estimate is on, which is the only thing the service acts on too.
 */
export const BAND_COPY = {
  unscored: {
    chip: 'Not scoring',
    detail: 'Waiting for enough clean tracking data.',
  },
  quiet: {
    chip: 'Below the reporting line',
    detail: 'The system stays quiet below this line.',
  },
  reporting: {
    chip: 'Above the reporting line',
    detail: 'Strong enough to be worth saying — if it holds and is not a repeat.',
  },
}

/** Stream states, and the sentence each one is allowed to claim. */
export const STREAM_COPY = {
  connecting: { label: 'connecting', tone: '', note: 'Opening the stream.' },
  live: { label: 'live', tone: 'ok', note: '' },
  reconnecting: { label: 'reconnecting', tone: 'warn', note: 'Lost the stream; retrying.' },
  finished: { label: 'finished', tone: '', note: 'The replay reached the end.' },
  offline: { label: 'offline', tone: 'bad', note: 'The service may have stopped.' },
}

/** Readiness has three states, so it needs three tones — not two. */
export const READINESS = {
  unknown: { label: 'checking', tone: '' },
  ready: { label: 'ready', tone: 'ok' },
  notReady: { label: 'not ready', tone: 'bad' },
}

/** Which readiness entry applies, with `null` meaning "no answer yet". */
export function readinessOf(ready) {
  if (ready === null || ready === undefined) return READINESS.unknown
  return ready ? READINESS.ready : READINESS.notReady
}

/**
 * Human wording for each `SuppressionReason` the service can report.
 *
 * The keys mirror the closed enum in `insight/types.py`, which is also what the
 * Prometheus labels use. An unrecognised key is shown verbatim rather than
 * dropped: a reason the UI has not been taught about is exactly the thing an
 * engineer needs to see, and silently hiding it would make a deployment
 * mismatch invisible.
 */
const SUPPRESSION_LABELS = {
  low_confidence: 'Below the reporting line',
  invalid_window: 'Window not usable',
  insufficient_frames: 'Not enough frames yet',
  cooldown: 'Too soon after the last one',
  duplicate_recent: 'Same thing, said recently',
  stale_situation: 'Moment had already passed',
  already_in_box: 'Ball already in the box',
  model_unavailable: 'No model loaded',
  schema_mismatch: 'Feature schema mismatch',
  not_yet_sustained: 'Not sustained yet',
  dead_ball: 'Ball out of play',
}

/** Label for a suppression reason, falling back to the raw wire value. */
export function suppressionLabel(reason) {
  if (!reason) return ABSENT
  return SUPPRESSION_LABELS[reason] ?? reason
}

/**
 * Stable identity for an insight.
 *
 * Index-based keys re-key every card when a new insight is prepended, which
 * remounts the whole list. These three fields also make a natural dedupe key
 * when the history fetched from `GET /insights` is merged with events that
 * arrived live: the per-kind cooldown means two insights cannot share all three.
 */
export function keyFor(insight) {
  return `${insight.period}-${insight.match_time_s}-${insight.kind}`
}

/** Dead band for the trend readout: below this, call it steady. */
const TREND_DEAD_BAND = 0.05

/**
 * Direction of travel across a series that may contain gaps.
 *
 * Compares the first and last *scored* points, because a declined window is not
 * a low estimate and must not be read as one.
 */
export function trendOf(points) {
  const scored = points.filter((p) => p !== null && p !== undefined)
  if (scored.length < 2) return null
  const delta = scored[scored.length - 1] - scored[0]
  if (delta > TREND_DEAD_BAND) return 'rising'
  if (delta < -TREND_DEAD_BAND) return 'falling'
  return 'steady'
}
