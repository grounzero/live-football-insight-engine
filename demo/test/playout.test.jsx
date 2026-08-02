/**
 * Unit tests for the playout buffer and its render clock.
 *
 * These are the correctness half of the smoothing work: the rendering half is a
 * question about a real canvas at a real refresh rate and is answered by
 * watching one. What can be pinned down here is that the buffer orders frames by
 * source time rather than arrival, that a blend at the midpoint is the midpoint,
 * that no blend ever crosses a barrier, and that the clock cannot run backwards
 * — which is the failure a viewer would actually notice.
 *
 * Built with vite (`npm run test:ssr`) and run with `node --test`, the same way
 * `layout.test.jsx` is. Nothing extra is installed for it.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  DEFAULT_PLAYOUT_DELAY_MS,
  MAX_EXTRAPOLATION_MS,
  MAX_RECOVERY_RATE,
  createPlayout,
} from '../src/playout.js'

/** A frame shaped like the service's, with two players a side. */
function frame(t, { lap = 0, period = 1, fixture = 'Synthetic_Demo', speed = 4, x = 0 } = {}) {
  return {
    match_time_s: t,
    frame: Math.round(t * 25),
    lap,
    period,
    fixture,
    speed,
    home: [
      [x, 0],
      [x + 10, 5],
    ],
    away: [
      [-x, 1],
      [-x - 10, -5],
    ],
    ball: [x * 2, x],
    probability: 0.4,
    window_valid: true,
    attacking_team: 'home',
    attacking_right: true,
    suppression: 'low_confidence',
  }
}

/** A buffer holding `samples`, pushed in the order given. */
function bufferAt(samples) {
  const playout = createPlayout()
  for (const s of samples) playout.push(s)
  return playout
}

/**
 * Positions are floats, so they are compared with a tolerance.
 *
 * `(1.1 - 1.0) / (1.2 - 1.0)` is 0.5000000000000002 in binary floating point,
 * and an exact comparison would be asserting the rounding rather than the
 * interpolation. Discrete values — ids, laps, teams, reasons — are still
 * compared exactly, everywhere below.
 */
function assertClose(actual, expected, what) {
  assert.equal(actual.length, expected.length, `${what}: wrong arity`)
  for (let i = 0; i < expected.length; i += 1) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) < 1e-9,
      `${what}: ${actual[i]} is not ${expected[i]}`,
    )
  }
}

test('frames are ordered by source time, not arrival order', () => {
  // Deliberately non-monotonic in x, so a buffer that appended 1.1 at the end
  // instead of inserting it produces a visibly different blend rather than
  // coincidentally the same one.
  const playout = bufferAt([frame(1.0, { x: 0 }), frame(1.2, { x: 20 }), frame(1.1, { x: 100 })])

  // Correctly ordered, 1.15 blends 1.1 -> 1.2, halfway between x=100 and x=20.
  // Appended, it would blend 1.0 -> 1.2 and land on x=15.
  const mid = playout.sampleAt(1.15)
  assert.equal(mid.interpolated, true)
  assertClose(mid.home[0], [60, 0], 'midpoint of the reordered pair')
})

test('a duplicate source time is ignored rather than inserted twice', () => {
  const playout = createPlayout()
  assert.equal(playout.push(frame(2.0)), true)
  assert.equal(playout.push(frame(2.0)), false)
  assert.equal(playout.stats().depth, 1)
  assert.equal(playout.stats().dropped, 1)
})

test('a frame the clock has already passed is discarded', () => {
  const playout = createPlayout()
  playout.push(frame(10.0))
  playout.push(frame(10.5))
  // Run the clock well past the old frame, then offer one from behind it.
  for (let i = 0; i < 100; i += 1) playout.advance(0.05)
  assert.equal(playout.push(frame(9.0)), false)
})

test('a malformed sample is refused without disturbing the buffer', () => {
  const playout = bufferAt([frame(1.0)])
  assert.equal(playout.push(null), false)
  assert.equal(playout.push({ home: [] }), false)
  assert.equal(playout.stats().depth, 1)
})

test('the midpoint of two frames is the midpoint of their positions', () => {
  const playout = bufferAt([frame(1.0, { x: 0 }), frame(1.2, { x: 10 })])

  const mid = playout.sampleAt(1.1)
  assertClose(mid.home[0], [5, 0], 'home 0')
  assertClose(mid.home[1], [15, 5], 'home 1')
  assertClose(mid.away[0], [-5, 1], 'away 0')
  assertClose(mid.ball, [10, 5], 'ball')
})

test('everything the server decided is a step value, never blended', () => {
  const a = frame(1.0)
  const b = { ...frame(1.2), probability: 0.9, attacking_team: 'away', suppression: 'cooldown' }
  const mid = createPlayout()
  mid.push(a)
  mid.push(b)

  const sample = mid.sampleAt(1.1)
  // A blended probability is a number no model produced. The earlier frame's
  // decision holds until the later frame is genuinely reached.
  assert.equal(sample.probability, 0.4)
  assert.equal(sample.attacking_team, 'home')
  assert.equal(sample.suppression, 'low_confidence')
})

test('an interpolated sample is marked and leaves the stored frames untouched', () => {
  const source = frame(1.0)
  const playout = bufferAt([source, frame(1.2, { x: 10 })])

  const sample = playout.sampleAt(1.1)
  assert.equal(sample.interpolated, true)
  // Nothing in the demo posts a frame to an inference endpoint, and the marker
  // is what would make it obvious if anything ever did. The source frame must
  // also survive the blend unmutated, or the next interpolation drifts.
  assert.deepEqual(source.home[0], [0, 0])
  assert.equal(source.interpolated, undefined)
  assert.equal(playout.sampleAt(1.0).interpolated, undefined)
})

test('a player missing at either end of a blend stays missing', () => {
  const a = frame(1.0)
  const b = frame(1.2, { x: 10 })
  a.home = [null, a.home[1]]
  b.home = [b.home[0], null]

  const sample = bufferAt([a, b]).sampleAt(1.1)
  assert.equal(sample.home[0], null, 'a player who appears mid-blend is not conjured')
  assert.equal(sample.home[1], null, 'a player who disappears mid-blend is not held')
})

test('a missing ball does not become a position', () => {
  const a = { ...frame(1.0), ball: null }
  const sample = bufferAt([a, frame(1.2)]).sampleAt(1.1)
  assert.equal(sample.ball, null)
})

for (const [label, later] of [
  ['lap', frame(1.2, { lap: 1 })],
  ['period', frame(1.2, { period: 2 })],
  ['fixture', frame(1.2, { fixture: 'Synthetic_Demo_Counter' })],
]) {
  test(`no blend crosses a ${label} barrier`, () => {
    const playout = bufferAt([frame(1.0, { x: 0 }), { ...later, home: [[100, 0], [110, 5]] }])

    // Held, not blended: the earlier position stays on screen until the clock
    // genuinely reaches the far side.
    const held = playout.sampleAt(1.1)
    assert.deepEqual(held.home[0], [0, 0])
    assert.equal(held.interpolated, undefined)

    const after = playout.sampleAt(1.25)
    assert.deepEqual(after.home[0], [100, 0])
  })
}

test('extrapolation past the newest frame is bounded, then freezes', () => {
  const playout = createPlayout()
  playout.push(frame(1.0, { x: 0 }))
  playout.push(frame(1.2, { x: 10 }))

  // The clock is what enforces the cap, so drive it rather than sampling
  // arbitrary times: at 4x, 80 ms of wall clock is 0.32 s of match time.
  for (let i = 0; i < 200; i += 1) playout.advance(0.05)
  const ceiling = 1.2 + (MAX_EXTRAPOLATION_MS / 1000) * 4
  assert.ok(playout.renderTime() <= ceiling + 1e-9, 'clock ran past the extrapolation cap')

  const frozen = playout.sampleAt(playout.renderTime())
  const again = playout.sampleAt(playout.renderTime())
  assert.deepEqual(frozen.home[0], again.home[0], 'a frozen display must be stable')
})

test('extrapolation never runs past a barrier', () => {
  const playout = bufferAt([frame(1.0, { x: 0 }), frame(1.2, { lap: 1, x: 10 })])

  // The newest two frames straddle a lap, so there is no velocity to continue.
  const beyond = playout.sampleAt(1.4)
  assert.deepEqual(beyond.home[0], [10, 0])
  assert.equal(beyond.interpolated, undefined)
})

test('the render clock never moves backwards', () => {
  const playout = createPlayout()
  playout.push(frame(1.0))

  let previous = playout.renderTime()
  for (let i = 0; i < 300; i += 1) {
    // A stream that stalls, resumes, and stalls again.
    if (i === 40) playout.push(frame(1.2))
    if (i === 45) playout.push(frame(1.4))
    if (i === 200) playout.push(frame(3.0))
    playout.advance(0.016)
    const now = playout.renderTime()
    assert.ok(now >= previous, `clock went backwards at tick ${i}: ${previous} -> ${now}`)
    previous = now
  }
})

test('the clock starts a full delay behind the first frame', () => {
  const playout = createPlayout()
  playout.push(frame(10.0, { speed: 4 }))

  // 180 ms of wall clock at 4x is 0.72 s of match time.
  const expected = 10.0 - (DEFAULT_PLAYOUT_DELAY_MS / 1000) * 4
  assert.ok(Math.abs(playout.renderTime() - expected) < 1e-9)
  // And the first frame is what is drawn until the clock reaches it, rather
  // than a blank pitch.
  assert.deepEqual(playout.sampleAt(playout.renderTime()).home[0], [0, 0])
})

test('recovery is gradual rather than a jump to the newest frame', () => {
  const playout = createPlayout()
  playout.push(frame(0.0))
  // A long gap, then a burst: the classic Railway pattern.
  for (let i = 0; i < 30; i += 1) playout.advance(0.016)
  const before = playout.renderTime()
  for (let i = 1; i <= 40; i += 1) playout.push(frame(i * 0.05))

  playout.advance(0.016)
  const step = playout.renderTime() - before
  const speed = 4
  assert.ok(step > 0, 'the clock must keep moving')
  assert.ok(
    step <= 0.016 * speed * MAX_RECOVERY_RATE + 1e-9,
    `clock jumped ${step}s in one 16 ms tick`,
  )
  assert.ok(playout.renderTime() < 2.0, 'the clock snapped to the newest frame')
})

test('a 500 ms outage freezes and then recovers without moving backwards', () => {
  const playout = createPlayout()
  for (let i = 0; i <= 10; i += 1) playout.push(frame(i * 0.05))

  const seen = []
  // 500 ms of wall clock with nothing arriving.
  for (let i = 0; i < 31; i += 1) {
    playout.advance(0.016)
    seen.push(playout.renderTime())
  }
  assert.ok(playout.stats().underruns > 0, 'a 500 ms outage must be reported as an underrun')

  const frozenAt = playout.renderTime()
  for (let i = 11; i <= 40; i += 1) playout.push(frame(i * 0.05))
  for (let i = 0; i < 30; i += 1) {
    playout.advance(0.016)
    seen.push(playout.renderTime())
  }

  assert.ok(playout.renderTime() > frozenAt, 'the clock must resume after the buffer refills')
  for (let i = 1; i < seen.length; i += 1) {
    assert.ok(seen[i] >= seen[i - 1], 'the clock moved backwards during recovery')
  }
})

test('reset clears the buffer and unsets the clock', () => {
  const playout = bufferAt([frame(1.0), frame(1.2)])
  playout.reset('fixture')

  assert.equal(playout.renderTime(), null)
  assert.equal(playout.sampleAt(1.1), null)
  assert.equal(playout.stats().depth, 0)
  assert.equal(playout.stats().lastReset, 'fixture')

  // The next frame decides where the clock restarts, which is the only value
  // that cannot produce a backwards jump.
  playout.push(frame(50.0))
  assert.ok(playout.renderTime() < 50.0)
  assert.deepEqual(playout.sampleAt(playout.renderTime()).home[0], [0, 0])
})

test('the buffer stays bounded under a stream that is never sampled', () => {
  const playout = createPlayout()
  for (let i = 0; i < 5000; i += 1) playout.push(frame(i * 0.05))
  assert.ok(playout.stats().depth <= 120, `buffer grew to ${playout.stats().depth}`)
})

test('stats report the lag in wall milliseconds, not match seconds', () => {
  const playout = createPlayout()
  playout.push(frame(10.0, { speed: 4 }))
  const stats = playout.stats()

  // The clock starts one delay behind, so the reported lag is that delay —
  // in wall time, whatever the replay speed.
  assert.ok(Math.abs(stats.lagMs - DEFAULT_PLAYOUT_DELAY_MS) < 1e-6)
  assert.equal(stats.speed, 4)
  assert.equal(stats.targetLagMs, DEFAULT_PLAYOUT_DELAY_MS)
})
