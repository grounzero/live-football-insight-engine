"""Live football insight engine.

Replays recorded tracking data as though it were arriving live, predicts whether
the attacking team will enter the opposition penalty area within a short horizon,
and converts that prediction into a qualified, viewer-facing insight.

The package is organised as a pipeline, and each stage is independently testable:

``data``
    Acquisition, parsing of both Metrica formats, validation, orientation.
``labels``
    Penalty-area-entry target construction and episode grouping.
``features``
    Identity-invariant spatial and temporal features, built through a causal
    event view that cannot see the future.
``models``
    Baselines, the temporal model, evaluation and ONNX export.
``insight``
    Candidate construction and the editorial layer that decides what a viewer
    actually sees.
``replay`` / ``serving``
    Deterministic live replay and the production-style service.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
