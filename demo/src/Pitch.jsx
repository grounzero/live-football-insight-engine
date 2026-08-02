import { useCallback, useEffect, useMemo, useRef } from 'react'

import { LEGEND, palette } from './theme.js'
import { formatClock } from './format.js'

// Canonical pitch coordinates: metres, origin at the centre spot,
// x in [-52.5, 52.5], y in [-34, 34] with +y toward the top touchline.
const LENGTH = 105
const WIDTH = 68
const BOX_DEPTH = 16.5
const BOX_WIDTH = 40.32
const SIX_DEPTH = 5.5
const SIX_WIDTH = 18.32
const GOAL_WIDTH = 7.32
const GOAL_DEPTH = 1.8
const CENTRE_CIRCLE = 9.15

/**
 * Size the backing store for the device pixel ratio and return a projection.
 *
 * Returns the CSS-pixel dimensions and a metre-to-pixel projection, so nothing
 * downstream has to know about the pixel ratio again.
 *
 * Called on mount and on resize, not per frame. Reading `clientWidth` is a
 * layout query, and the draw loop now runs at the display's refresh rate rather
 * than the network's — asking the browser to measure the page sixty times a
 * second to discover a number that changes when someone drags a window is work
 * for nothing.
 */
function measureCanvas(canvas) {
  const ctx = canvas.getContext('2d')
  const dpr = window.devicePixelRatio || 1
  const cssWidth = canvas.clientWidth
  const cssHeight = Math.round((cssWidth * WIDTH) / LENGTH)
  if (canvas.width !== cssWidth * dpr || canvas.height !== cssHeight * dpr) {
    canvas.width = cssWidth * dpr
    canvas.height = cssHeight * dpr
    canvas.style.height = `${cssHeight}px`
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const scale = cssWidth / LENGTH
  return {
    ctx,
    cssWidth,
    cssHeight,
    scale,
    toX: (x) => (x + LENGTH / 2) * scale,
    toY: (y) => (WIDTH / 2 - y) * scale,
  }
}

/** Draw the turf, touchlines, halfway line and centre circle. */
function drawSurface({ ctx, cssWidth, cssHeight, scale, toX, toY }) {
  ctx.fillStyle = palette().turf
  ctx.fillRect(0, 0, cssWidth, cssHeight)

  ctx.strokeStyle = palette().line
  ctx.lineWidth = 1.2
  ctx.strokeRect(toX(-LENGTH / 2), toY(WIDTH / 2), LENGTH * scale, WIDTH * scale)

  ctx.beginPath()
  ctx.moveTo(toX(0), toY(WIDTH / 2))
  ctx.lineTo(toX(0), toY(-WIDTH / 2))
  ctx.stroke()

  ctx.beginPath()
  ctx.arc(toX(0), toY(0), CENTRE_CIRCLE * scale, 0, Math.PI * 2)
  ctx.stroke()
}

/** Draw the penalty area, six-yard box and goal frame at both ends. */
function drawGoalAreas({ ctx, scale, toX, toY }) {
  for (const side of [-1, 1]) {
    const boxX = side === 1 ? LENGTH / 2 - BOX_DEPTH : -LENGTH / 2
    ctx.strokeRect(toX(boxX), toY(BOX_WIDTH / 2), BOX_DEPTH * scale, BOX_WIDTH * scale)
    const sixX = side === 1 ? LENGTH / 2 - SIX_DEPTH : -LENGTH / 2
    ctx.strokeRect(toX(sixX), toY(SIX_WIDTH / 2), SIX_DEPTH * scale, SIX_WIDTH * scale)
    const goalX = side === 1 ? LENGTH / 2 : -LENGTH / 2 - GOAL_DEPTH
    ctx.strokeRect(toX(goalX), toY(GOAL_WIDTH / 2), GOAL_DEPTH * scale, GOAL_WIDTH * scale)
  }
}

/**
 * Shade the penalty area currently being attacked.
 *
 * That box is the target of the prediction, so it is worth making obvious. It
 * stayed unshaded for a long time because the direction was never sent to the
 * browser, which left the whole screen describing a forecast without ever
 * showing what was being forecast.
 */
function drawAttackingBox({ ctx, scale, toX, toY }, attackingRight) {
  if (attackingRight === null || attackingRight === undefined) return
  const boxX = attackingRight ? LENGTH / 2 - BOX_DEPTH : -LENGTH / 2
  ctx.fillStyle = palette().shade
  ctx.fillRect(toX(boxX), toY(BOX_WIDTH / 2), BOX_DEPTH * scale, BOX_WIDTH * scale)
}

/**
 * Draw a direction marker along the halfway line.
 *
 * The shaded box says which end is the target; this says which way play is
 * running, which is what someone joining mid-passage needs first. It sits near
 * the halfway line rather than over the players so it never occludes them, and
 * it only ever reinforces the wording in the transport bar — the text is the
 * primary cue, this is the glance.
 */
function drawDirection({ ctx, scale, toX, toY }, attackingRight) {
  if (attackingRight === null || attackingRight === undefined) return
  const sign = attackingRight ? 1 : -1
  const y = toY(WIDTH / 2 - 4)
  const from = toX(-7 * sign)
  const to = toX(7 * sign)
  const head = 2.2 * scale

  ctx.strokeStyle = palette().line
  ctx.lineWidth = 1.6
  ctx.beginPath()
  ctx.moveTo(from, y)
  ctx.lineTo(to, y)
  ctx.moveTo(to, y)
  ctx.lineTo(to - head * sign, y - head * 0.6)
  ctx.moveTo(to, y)
  ctx.lineTo(to - head * sign, y + head * 0.6)
  ctx.stroke()
}

/** Draw one team's players, with column 0 shown as the keeper. */
function drawTeam({ ctx, scale, toX, toY }, positions, colour) {
  positions.forEach((p, index) => {
    if (!p) return
    ctx.beginPath()
    ctx.arc(toX(p[0]), toY(p[1]), Math.max(3, 0.9 * scale), 0, Math.PI * 2)
    ctx.fillStyle = index === 0 ? palette().keeper : colour
    ctx.fill()
  })
}

/** Draw the ball, outlined so it stays visible against a light shirt. */
function drawBall({ ctx, scale, toX, toY }, ball) {
  if (!ball) return
  ctx.beginPath()
  ctx.arc(toX(ball[0]), toY(ball[1]), Math.max(2.5, 0.55 * scale), 0, Math.PI * 2)
  ctx.fillStyle = palette().ball
  ctx.fill()
  ctx.strokeStyle = 'rgba(0,0,0,0.55)'
  ctx.lineWidth = 1
  ctx.stroke()
}

/** Which way the named team is playing, in words. */
export function directionText(frame) {
  if (!frame?.attacking_team || frame.attacking_right === null) return null
  const team = frame.attacking_team === 'home' ? 'Home' : 'Away'
  return `${team} attacking ${frame.attacking_right ? 'right' : 'left'}`
}

/** One sentence describing the pitch, for readers who cannot see it. */
function pitchLabel(second, period, direction) {
  if (second === null) return 'Pitch. Waiting for the first frame.'
  const where = direction ? ` ${direction}.` : ''
  return `Pitch. ${formatClock(second)}, period ${period}.${where}`
}

/**
 * The colour key for the pitch.
 *
 * Colour alone is not a legend: nothing else on the page explains that blue is
 * the home side or that the pale box is the one the model is predicting entry
 * into. The swatches read the same custom properties the canvas does, so the
 * key and the drawing cannot disagree.
 */
export function Legend() {
  return (
    <ul className="legend">
      {LEGEND.map(([token, label]) => (
        <li key={token}>
          <span className="swatch" style={{ background: `var(${token})` }} aria-hidden="true" />
          {label}
        </li>
      ))}
    </ul>
  )
}

/**
 * Draws the pitch, players and ball on a canvas.
 *
 * Drawing is driven by `requestAnimationFrame` and reads from the playout
 * buffer, not from React state on message arrival. That inversion is the whole
 * point. The server publishes about twenty pictures a second and the network
 * delivers them unevenly; a canvas redrawn on arrival inherits that unevenness
 * exactly, and over the public internet it reads as the pitch freezing and then
 * lurching. Sampling a buffered, timestamped stream on the display's own clock
 * decouples the two, and the positions between real frames are interpolated
 * rather than invented — see `playout.js`.
 *
 * @param {object} props
 * @param {object} props.playout The shared playout buffer, a stable object.
 * @param {object|null} props.frame The most recent frame as React state, used
 *   only for the text description. It is deliberately throttled far below the
 *   draw rate: this is a label, not an animation.
 */
export default function Pitch({ playout, frame }) {
  const canvasRef = useRef(null)
  const viewRef = useRef(null)

  const draw = useCallback((sample) => {
    const view = viewRef.current
    if (!view) return

    view.ctx.clearRect(0, 0, view.cssWidth, view.cssHeight)
    const attackingRight = sample?.attacking_right ?? null

    drawSurface(view)
    drawGoalAreas(view)
    drawAttackingBox(view, attackingRight)
    drawDirection(view, attackingRight)

    if (!sample) return
    drawTeam(view, sample.home, palette().home)
    drawTeam(view, sample.away, palette().away)
    drawBall(view, sample.ball)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined

    viewRef.current = measureCanvas(canvas)
    // The canvas is sized from its CSS width, and nothing recomputes that on its
    // own. The draw loop below no longer measures, so this is the only thing
    // that notices a resize — including while the replay is paused, finished or
    // offline, which is exactly when someone resizes to look more closely.
    const observer = new ResizeObserver(() => {
      viewRef.current = measureCanvas(canvas)
    })
    observer.observe(canvas)

    let handle = 0
    let previous = null
    const tick = (now) => {
      handle = requestAnimationFrame(tick)
      // The first tick has no interval to advance over, and a tab returning from
      // the background reports one that spans the whole time it was hidden. The
      // clamp keeps that from being spent as replay time in a single step; the
      // buffer's own recovery limit then closes the remaining gap smoothly.
      if (previous !== null) playout.advance(Math.min((now - previous) / 1000, 0.25))
      previous = now
      draw(playout.sampleAt(playout.renderTime()))
    }
    handle = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(handle)
      observer.disconnect()
    }
  }, [draw, playout])

  // Recomputed about once a second, from state that updates a few times a
  // second — never from the draw loop. `role="img"` is not a live region, so
  // this is read on demand rather than announced on change; narrating sixty
  // position updates a second would be unusable.
  const second = frame ? Math.floor(frame.match_time_s) : null
  const period = frame?.period ?? null
  const direction = directionText(frame)
  const label = useMemo(() => pitchLabel(second, period, direction), [second, period, direction])

  return (
    <canvas ref={canvasRef} className="pitch" role="img" aria-label={label}>
      A diagram of player and ball positions on a football pitch. The insight feed carries the
      same analysis as text.
    </canvas>
  )
}
