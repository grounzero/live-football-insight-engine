# ADR 0006: One feature implementation for training and serving

**Status:** accepted · **Date:** 2026-08-01

## Context

Two implementations of "the same" features, one vectorised for training and one incremental for
serving, is the classic source of train/serve skew: the model evaluates well and quietly degrades
in production.

## Decision

The live path calls exactly the function the training path calls, over a shorter array. Three
details make that possible:

1. **Backward-difference velocities only.** Central differences would make a timestep depend on
   frames after it, so the same instant would differ depending on whether it sat mid-window
   (offline) or at the window's end (live).
2. **Velocity warm-up.** The rolling buffer holds `observation + span` frames and discards the
   leading `span` rows after computing features, so every retained row has real history rather than
   edge padding. The offline slice does the same.
3. **Attacking team fixed per window.** The live buffer has one attacking team, the one in
   possession at the prediction instant. So offline features are computed per
   `(period, attacking team)` over the whole period, and a window is sliced from the matrix of the
   team in possession at its final frame.

## Consequences

- Point 3 is the one that would have been missed. Computing each frame with whichever team happened
  to hold the ball at that frame is the natural implementation and is wrong; no test of the feature
  functions themselves would catch it.
- Per-frame possession context is recorded once when a frame arrives and never revised, so live
  values vary across a window exactly as the offline computation does.
- Cost: features are computed twice offline (once per attacking hypothesis). Cheap, and worth it.
