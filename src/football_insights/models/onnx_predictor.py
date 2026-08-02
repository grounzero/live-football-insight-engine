"""Serving an exported ONNX artifact.

Separate from :mod:`football_insights.models.export_onnx` because the two halves
have different dependencies. *Exporting* needs PyTorch; *scoring* needs only
ONNX Runtime. Keeping them in one module would have made torch an import-time
requirement of the serving path, and the published container installs the base
dependency set precisely to avoid carrying a training framework it never calls.

:mod:`export_onnx` re-exports both names, so existing callers are unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import onnxruntime as ort

    from football_insights.models.base import ModelMetadata


def load_session(path: Path) -> ort.InferenceSession:
    """Open an ONNX Runtime session pinned to CPU.

    CPU is the target: inference must fit inside a live latency budget where a
    device transfer would cost more than the forward pass itself.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    # The dynamo exporter records the traced output shape even though the batch
    # axis is dynamic, so ORT warns on every call with a batch other than the
    # traced one. Results are correct — parity is asserted in CI — so the
    # warning is suppressed rather than allowed to flood service logs.
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class OnnxPredictor:
    """An exported model served through ONNX Runtime."""

    def __init__(self, path: Path, metadata: ModelMetadata) -> None:
        """Open a session for an exported artifact.

        Args:
            path: Path to the ``.onnx`` file.
            metadata: The model's :class:`ModelMetadata`.
        """
        self._session = load_session(path)
        self._metadata = metadata
        self._input = self._session.get_inputs()[0].name

    @property
    def metadata(self) -> ModelMetadata:
        """Identity and provenance, shared with the PyTorch artifact."""
        return self._metadata

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Score a batch of windows.

        Args:
            windows: Array ``(n, sequence_length, n_features)``.

        Returns:
            Probabilities of shape ``(n,)``.
        """
        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        output = self._session.run(None, {self._input: arr})[0]
        return np.asarray(output).reshape(-1).astype(np.float64)
