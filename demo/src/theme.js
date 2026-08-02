/**
 * The canvas palette, read from the stylesheet.
 *
 * A canvas cannot use `var()`, so its colours would otherwise be a second copy
 * of the palette that drifts the moment either file is edited — the pitch's
 * home colour and `--accent` were already the same literal in two places, with
 * nothing to notice if one of them changed. Reading the custom properties makes
 * `styles.css` the single source, without asking the stylesheet to be built
 * from JavaScript, which would flash untokenised markup before React mounts.
 */

const CANVAS_TOKENS = {
  turf: ['--pitch-turf', '#12312a'],
  line: ['--pitch-line', 'rgba(255,255,255,0.45)'],
  home: ['--team-home', '#4da3ff'],
  away: ['--team-away', '#ff8f52'],
  ball: ['--pitch-ball', '#ffffff'],
  keeper: ['--team-keeper', '#ffe066'],
  shade: ['--pitch-shade', 'rgba(255,255,255,0.07)'],
}

let cached = null

/**
 * Resolve the canvas palette once and reuse it.
 *
 * Cached because `getComputedStyle` forces a style recalculation and the pitch
 * is redrawn twelve and a half times a second. There is a single theme, fixed
 * by `color-scheme: dark`, so there is nothing to invalidate.
 *
 * Resolved lazily on first draw rather than at module load, so the read happens
 * after the stylesheet has been applied regardless of import order. The literal
 * fallbacks are not decoration: a missing custom property reads as the empty
 * string, and assigning that to `fillStyle` is silently ignored, which would
 * draw the shape in whatever colour happened to be set last.
 */
export function palette() {
  if (cached) return cached
  const root = getComputedStyle(document.documentElement)
  cached = Object.freeze(
    Object.fromEntries(
      Object.entries(CANVAS_TOKENS).map(([name, [property, fallback]]) => [
        name,
        root.getPropertyValue(property).trim() || fallback,
      ]),
    ),
  )
  return cached
}

/** The legend's entries, in the order they are drawn on the pitch. */
export const LEGEND = [
  ['--team-home', 'Home'],
  ['--team-away', 'Away'],
  ['--team-keeper', 'Goalkeeper'],
  ['--pitch-ball', 'Ball'],
  ['--pitch-shade', 'Target penalty area'],
]
