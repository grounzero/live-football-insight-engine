import { useEffect, useRef } from 'react'

// Canonical pitch coordinates: metres, origin at the centre spot,
// x in [-52.5, 52.5], y in [-34, 34] with +y toward the top touchline.
const LENGTH = 105
const WIDTH = 68
const BOX_DEPTH = 16.5
const BOX_WIDTH = 40.32
const SIX_DEPTH = 5.5
const SIX_WIDTH = 18.32
const GOAL_WIDTH = 7.32

const COLOURS = {
  turf: '#12312a',
  line: 'rgba(255,255,255,0.45)',
  home: '#4da3ff',
  away: '#ff8f52',
  ball: '#ffffff',
  keeper: '#ffe066',
}

/**
 * Draws the pitch, players and ball on a canvas.
 *
 * The canvas is redrawn from the latest frame only; no animation state is kept,
 * so a dropped or late frame simply shows the previous position rather than
 * interpolating something that was never observed.
 */
export default function Pitch({ frame, attackingRight }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
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
    ctx.clearRect(0, 0, cssWidth, cssHeight)

    const scale = cssWidth / LENGTH
    const toX = (x) => (x + LENGTH / 2) * scale
    const toY = (y) => (WIDTH / 2 - y) * scale

    ctx.fillStyle = COLOURS.turf
    ctx.fillRect(0, 0, cssWidth, cssHeight)

    ctx.strokeStyle = COLOURS.line
    ctx.lineWidth = 1.2
    ctx.strokeRect(toX(-LENGTH / 2), toY(WIDTH / 2), LENGTH * scale, WIDTH * scale)

    ctx.beginPath()
    ctx.moveTo(toX(0), toY(WIDTH / 2))
    ctx.lineTo(toX(0), toY(-WIDTH / 2))
    ctx.stroke()

    ctx.beginPath()
    ctx.arc(toX(0), toY(0), 9.15 * scale, 0, Math.PI * 2)
    ctx.stroke()

    for (const side of [-1, 1]) {
      const boxX = side === 1 ? LENGTH / 2 - BOX_DEPTH : -LENGTH / 2
      ctx.strokeRect(toX(boxX), toY(BOX_WIDTH / 2), BOX_DEPTH * scale, BOX_WIDTH * scale)
      const sixX = side === 1 ? LENGTH / 2 - SIX_DEPTH : -LENGTH / 2
      ctx.strokeRect(toX(sixX), toY(SIX_WIDTH / 2), SIX_DEPTH * scale, SIX_WIDTH * scale)
      const goalX = side === 1 ? LENGTH / 2 : -LENGTH / 2 - 1.8
      ctx.strokeRect(toX(goalX), toY(GOAL_WIDTH / 2), 1.8 * scale, GOAL_WIDTH * scale)
    }

    // Shade the penalty area currently being attacked: the target of the
    // prediction, so it is worth making obvious.
    if (attackingRight !== null) {
      const boxX = attackingRight ? LENGTH / 2 - BOX_DEPTH : -LENGTH / 2
      ctx.fillStyle = 'rgba(255,255,255,0.07)'
      ctx.fillRect(toX(boxX), toY(BOX_WIDTH / 2), BOX_DEPTH * scale, BOX_WIDTH * scale)
    }

    if (!frame) return

    const drawTeam = (positions, colour) => {
      positions.forEach((p, index) => {
        if (!p) return
        ctx.beginPath()
        ctx.arc(toX(p[0]), toY(p[1]), Math.max(3, 0.9 * scale), 0, Math.PI * 2)
        ctx.fillStyle = index === 0 ? COLOURS.keeper : colour
        ctx.fill()
      })
    }
    drawTeam(frame.home, COLOURS.home)
    drawTeam(frame.away, COLOURS.away)

    if (frame.ball) {
      ctx.beginPath()
      ctx.arc(toX(frame.ball[0]), toY(frame.ball[1]), Math.max(2.5, 0.55 * scale), 0, Math.PI * 2)
      ctx.fillStyle = COLOURS.ball
      ctx.fill()
      ctx.strokeStyle = 'rgba(0,0,0,0.55)'
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }, [frame, attackingRight])

  return <canvas ref={canvasRef} className="pitch" />
}
