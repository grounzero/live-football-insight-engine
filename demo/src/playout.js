/**
 * Presentation-only playout: a short ordered buffer, a render clock, and linear
 * interpolation between the frames the server actually sent.
 *
 * The server publishes about twenty pictures a second on a wall clock. A display
 * refreshes sixty times a second, and the network delivers those twenty at
 * intervals that are even locally and are not even over the public internet —
 * measured against the hosted deployment, the median gap is 10 ms and the 99th
 * percentile is 87 ms. Drawing each frame the moment it arrives puts that
 * distribution straight onto the pitch: a long gap freezes, and the frames
 * queued behind it are applied faster than they can be painted, so the ball
 * jumps. That is what this module exists to stop.
 *
 * Nothing here is a model input. Interpolated positions are geometry for the
 * canvas and nothing else — every value the server computed (probability,
 * possession, direction, suppression) is carried through as a step function
 * taken from the earlier of the two real frames, never blended. A blended
 * probability would be a number no model produced, presented as though one had.
 * Interpolated samples are marked, so anything that did try to send one back
 * would be recognisable.
 */

/** Wall-clock milliseconds the render clock deliberately lags the newest frame. */
export const DEFAULT_PLAYOUT_DELAY_MS = 180

/**
 * Wall-clock milliseconds the render clock may run past the newest frame.
 *
 * Beyond this the display freezes on the last real position rather than
 * continuing to invent motion. A 500 ms outage cannot be hidden, and pretending
 * otherwise means showing half a second of movement that never happened and then
 * correcting it — which is more jarring than the pause.
 */
export const MAX_EXTRAPOLATION_MS = 80

/**
 * How much faster or slower than the replay the render clock may run.
 *
 * The clock drifts from its target lag whenever the network does something
 * unusual, and the correction has to be gradual: snapping to the newest frame
 * is the very jump this module removes. 1.15 recovers a 200 ms error in about a
 * second and a half, which is slow enough that nobody sees the speed change.
 */
export const MAX_RECOVERY_RATE = 1.15

/** Frames retained. At 20 Hz this is six seconds — far more than the delay needs. */
const CAPACITY = 120

/** Render-clock error tolerated before the rate is nudged, in match seconds. */
const LAG_TOLERANCE_S = 0.01

/**
 * Which replay a frame belongs to.
 *
 * Two frames may only be blended when all three agree. A lap wrap rewinds the
 * clock to zero, a fixture change swaps the whole match, and a period change
 * swaps the ends the teams attack — interpolating across any of them walks
 * twenty-two players across the pitch over a single 50 ms step.
 */
function barrierOf(sample) {
  return `${sample.fixture ?? ''}|${sample.lap ?? 0}|${sample.period ?? 0}`
}

/**
 * Positions blend elementwise; a player absent at either end stays absent.
 *
 * Absent means absent, not "use whichever end has one". Falling back to the
 * populated side would draw a player at a position the blend has no evidence
 * for, and would hold them there for the whole span — which is precisely the
 * stale-position bug the tracking data's own NaN handling exists to avoid.
 */
function lerpPositions(from, to, alpha) {
  if (!from || !to) return null
  const n = Math.max(from.length, to.length)
  const out = new Array(n)
  for (let i = 0; i < n; i += 1) {
    const p = from[i]
    const q = to[i]
    // Index is identity: column 0 is the keeper on both sides of the blend, so a
    // player who appears or disappears leaves a hole rather than shifting the
    // rest of the team by one.
    out[i] = p && q ? [p[0] + (q[0] - p[0]) * alpha, p[1] + (q[1] - p[1]) * alpha] : null
  }
  return out
}

/** One point blends the same way, with the same absence rule. */
function lerpPoint(from, to, alpha) {
  if (!from || !to) return null
  return [from[0] + (to[0] - from[0]) * alpha, from[1] + (to[1] - from[1]) * alpha]
}

/**
 * A frame for the canvas, built from a real one.
 *
 * Everything the server decided is inherited from `base` — the earlier of the
 * two frames — and only the geometry is replaced. `alpha` beyond [0, 1] is
 * extrapolation, which the caller has already bounded.
 */
function blend(base, next, alpha) {
  return {
    ...base,
    home: lerpPositions(base.home, next.home, alpha),
    away: lerpPositions(base.away, next.away, alpha),
    ball: lerpPoint(base.ball, next.ball, alpha),
    interpolated: true,
  }
}

/**
 * Create a playout buffer and its render clock.
 *
 * @param {object} [options]
 * @param {number} [options.delayMs] Wall-clock lag behind the newest frame.
 * @param {number} [options.maxExtrapolationMs] Wall-clock cap on running ahead.
 * @param {number} [options.maxRecoveryRate] Clock slew limit, as a multiple.
 * @param {number} [options.capacity] Frames retained.
 */
export function createPlayout({
  delayMs = DEFAULT_PLAYOUT_DELAY_MS,
  maxExtrapolationMs = MAX_EXTRAPOLATION_MS,
  maxRecoveryRate = MAX_RECOVERY_RATE,
  capacity = CAPACITY,
} = {}) {
  /** @type {object[]} ordered by source match time, oldest first. */
  let samples = []
  let renderT = null
  let underruns = 0
  let dropped = 0
  let received = 0
  let lastReset = null

  /** Match seconds per wall second. Every frame carries it; 1 until one does. */
  const speedOf = () => (samples.length ? (samples[samples.length - 1].speed ?? 1) : 1)

  const newestT = () => (samples.length ? samples[samples.length - 1].match_time_s : null)

  /**
   * Where the clock should sit: far enough behind the newest frame that an
   * ordinary network hiccup still has a frame on the far side to blend toward.
   * The delay is quoted in wall time and converted here, because the match
   * clock advances `speed` times faster than the wall does.
   */
  const targetT = () => {
    const newest = newestT()
    return newest === null ? null : newest - (delayMs / 1000) * speedOf()
  }

  /**
   * Discard frames the clock has already passed.
   *
   * One frame *before* the clock is deliberately kept: it is the anchor the
   * current position is interpolated from, and dropping it would leave nothing
   * to blend out of.
   */
  function trim() {
    if (renderT === null) {
      if (samples.length > capacity) {
        dropped += samples.length - capacity
        samples = samples.slice(-capacity)
      }
      return
    }
    let keepFrom = 0
    for (let i = 0; i < samples.length; i += 1) {
      if (samples[i].match_time_s <= renderT) keepFrom = i
      else break
    }
    if (keepFrom > 0) {
      dropped += keepFrom
      samples = samples.slice(keepFrom)
    }
    if (samples.length > capacity) {
      dropped += samples.length - capacity
      samples = samples.slice(-capacity)
    }
  }

  return {
    /**
     * Insert one frame from the network.
     *
     * Ordered by source time rather than by arrival, because arrival order is
     * exactly the thing that is not trustworthy. A duplicate frame id is
     * ignored, and a frame the clock has already gone past is discarded rather
     * than inserted behind it.
     */
    push(sample) {
      if (!sample || typeof sample.match_time_s !== 'number') return false
      received += 1

      // A new lap or fixture rewinds the source clock, so every frame of it is
      // "older" than where the render clock currently sits and the staleness
      // check below would reject all of them — the pitch would freeze for good.
      //
      // The barrier message says so too, and is published as critical precisely
      // so it cannot be dropped. Acting on the frame's own identity as well
      // costs one comparison and removes the dependence on two cadences
      // arriving in the right order, which is the kind of coupling that holds
      // until the day it does not. Period changes are deliberately excluded:
      // they keep the clock monotonic, so `sampleAt` can hold and then snap
      // without discarding a buffer that is still valid.
      const newest = samples[samples.length - 1]
      if (newest && (newest.lap !== sample.lap || newest.fixture !== sample.fixture)) {
        samples = []
        renderT = null
        lastReset = 'barrier'
      }

      if (renderT !== null && sample.match_time_s < renderT) {
        dropped += 1
        return false
      }

      let at = samples.length
      while (at > 0 && samples[at - 1].match_time_s > sample.match_time_s) at -= 1
      const neighbour = samples[at - 1]
      if (neighbour && neighbour.match_time_s === sample.match_time_s) {
        dropped += 1
        return false
      }
      samples.splice(at, 0, sample)

      if (renderT === null) {
        // Start a full delay behind the first frame, so the pitch holds that
        // position while the buffer fills instead of immediately running out of
        // frames to blend toward.
        renderT = sample.match_time_s - (delayMs / 1000) * (sample.speed ?? 1)
      }
      trim()
      return true
    },

    /**
     * Drop everything at a barrier.
     *
     * The clock is unset rather than rewound: the next frame decides where it
     * restarts, which is the only value that cannot produce a backwards jump.
     */
    reset(reason = 'unknown') {
      samples = []
      renderT = null
      lastReset = reason
      return reason
    },

    /**
     * Move the render clock forward by `dtSeconds` of wall time.
     *
     * Three rules, in order: never go backwards, never run more than the
     * extrapolation cap past the newest frame, and correct any drift from the
     * target lag by at most `maxRecoveryRate`. The last one is what keeps a
     * recovery from looking like a jump.
     */
    advance(dtSeconds) {
      if (renderT === null || !samples.length || !(dtSeconds > 0)) return renderT
      const speed = speedOf()
      const target = targetT()

      let rate = speed
      if (renderT < target - LAG_TOLERANCE_S) rate = speed * maxRecoveryRate
      else if (renderT > target + LAG_TOLERANCE_S) rate = speed / maxRecoveryRate

      const ceiling = newestT() + (maxExtrapolationMs / 1000) * speed
      const next = renderT + dtSeconds * rate
      if (next > newestT()) underruns += 1
      // `Math.max` before `Math.min`: if the clock is already past the ceiling —
      // which happens when a barrier shortens the buffer — it must stall there,
      // never step back.
      renderT = Math.max(renderT, Math.min(next, ceiling))
      trim()
      return renderT
    },

    /** The render clock, in source match seconds, or null before the first frame. */
    renderTime() {
      return renderT
    },

    /**
     * The frame to draw at source time `t`.
     *
     * @returns {object|null} A frame-shaped object, or null when nothing has
     *   arrived yet. Interpolated results carry `interpolated: true`.
     */
    sampleAt(t) {
      if (!samples.length) return null
      if (typeof t !== 'number') return samples[samples.length - 1]
      if (t <= samples[0].match_time_s) return samples[0]

      for (let i = 0; i < samples.length - 1; i += 1) {
        const a = samples[i]
        const b = samples[i + 1]
        if (t >= b.match_time_s) continue
        // Hold rather than blend across a barrier. `a` stays on screen until the
        // clock reaches `b`, at which point the loop above hands back `b` whole.
        if (barrierOf(a) !== barrierOf(b)) return a
        const span = b.match_time_s - a.match_time_s
        return span > 0 ? blend(a, b, (t - a.match_time_s) / span) : a
      }

      const last = samples[samples.length - 1]
      const previous = samples[samples.length - 2]
      if (!previous || barrierOf(previous) !== barrierOf(last)) return last
      const span = last.match_time_s - previous.match_time_s
      if (!(span > 0)) return last
      // Past the newest frame, within the cap the clock enforces: continue at
      // the velocity the last two frames actually described.
      return blend(previous, last, (t - previous.match_time_s) / span)
    },

    /** Counters for the diagnostics panel. Cheap, and read at most a few times a second. */
    stats() {
      const newest = newestT()
      return {
        depth: samples.length,
        received,
        dropped,
        underruns,
        lastReset,
        renderTime: renderT,
        newestTime: newest,
        // Reported in wall milliseconds because that is the unit the delay is
        // configured in; the match clock runs `speed` times faster.
        lagMs: renderT === null || newest === null ? null : ((newest - renderT) / speedOf()) * 1000,
        targetLagMs: delayMs,
        speed: speedOf(),
      }
    },
  }
}
