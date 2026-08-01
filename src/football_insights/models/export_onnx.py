"""ONNX export, parity checking and runtime benchmarking.

Exporting is only half of it. A graph that loads is not evidence that it
computes the same thing, so :func:`check_parity` compares the two runtimes on
real windows and reports the largest disagreement rather than asserting quietly.

Standardisation is folded into the exported graph as a constant. If it were left
outside, the ONNX artifact would be silently incomplete: anyone loading it
without also loading ``mean`` and ``scale`` would get plausible-looking
nonsense, which is exactly the sort of failure that survives to production.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn

from football_insights.types import JsonDict

if TYPE_CHECKING:
    from football_insights.models.temporal import TemporalPredictor

#: Maximum absolute probability difference tolerated between runtimes.
#: Float32 GRU accumulation differs slightly between implementations; measured
#: parity is far below this, and the benchmark report records the actual figure.
PARITY_TOLERANCE = 1e-4

#: Opset 18. The dynamo exporter converts down from 18 with a warning if asked
#: for less, so this avoids a needless conversion step.
OPSET_VERSION = 18


class _ExportWrapper(nn.Module):
    """Standardisation, network and sigmoid as one graph.

    The exported model takes raw features and returns a probability, so the
    ONNX artifact is self-contained and cannot be used incorrectly.
    """

    def __init__(self, model: nn.Module, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(np.asarray(mean, dtype=np.float32)))
        self.register_buffer("scale", torch.from_numpy(np.asarray(scale, dtype=np.float32)))

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """Score raw feature windows.

        Args:
            window: Tensor ``(batch, sequence_length, n_features)``.

        Returns:
            Probabilities of shape ``(batch, 1)``.
        """
        # ``self.mean``/``self.scale`` are registered buffers; nn.Module's
        # __getattr__ is typed as returning Module, so rebind them as tensors.
        mean = cast("torch.Tensor", self.mean)
        scale = cast("torch.Tensor", self.scale)
        standardised = (window - mean) / scale
        probability: torch.Tensor = torch.sigmoid(self.model(standardised))
        return probability


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency distribution for one runtime, in milliseconds."""

    runtime: str
    batch_size: int
    iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_per_s: float
    cold_start_ms: float

    def to_dict(self) -> JsonDict:
        """Serialisable form."""
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def export(
    predictor: TemporalPredictor,
    path: Path,
    sequence_length: int | None = None,
) -> Path:
    """Export a trained temporal model to ONNX.

    Args:
        predictor: The trained model.
        path: Destination ``.onnx`` path.
        sequence_length: Sequence length to trace with; taken from metadata when
            omitted.

    Returns:
        The written path.
    """
    metadata = predictor.metadata
    length = sequence_length or metadata.sequence_length
    wrapper = _ExportWrapper(
        predictor.module.cpu().eval(),
        predictor.standardiser.mean,
        predictor.standardiser.scale,
    ).eval()

    example = torch.zeros(1, length, metadata.n_features, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        input_names=["window"],
        output_names=["probability"],
        dynamic_axes={"window": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=OPSET_VERSION,
        dynamo=True,
    )
    return path


def load_session(path: Path) -> object:
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

    def __init__(self, path: Path, metadata: object) -> None:
        """Open a session for an exported artifact.

        Args:
            path: Path to the ``.onnx`` file.
            metadata: The model's :class:`ModelMetadata`.
        """
        self._session = load_session(path)
        self._metadata = metadata
        self._input = self._session.get_inputs()[0].name  # type: ignore[attr-defined]

    @property
    def metadata(self) -> object:
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
        output = self._session.run(None, {self._input: arr})[0]  # type: ignore[attr-defined]
        return np.asarray(output).reshape(-1).astype(np.float64)


def check_parity(
    predictor: TemporalPredictor,
    onnx_path: Path,
    windows: np.ndarray,
    tolerance: float = PARITY_TOLERANCE,
) -> JsonDict:
    """Compare PyTorch and ONNX Runtime on the same inputs.

    Args:
        predictor: The PyTorch model.
        onnx_path: The exported artifact.
        windows: Real windows to compare on.
        tolerance: Maximum absolute difference treated as agreement.

    Returns:
        A report with the measured maximum and mean absolute differences, and
        whether they are within tolerance.
    """
    reference = predictor.predict_proba(windows)
    exported = OnnxPredictor(onnx_path, predictor.metadata).predict_proba(windows)
    difference = np.abs(reference - exported)
    return {
        "samples": len(reference),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "tolerance": tolerance,
        "within_tolerance": bool(difference.max() <= tolerance),
        "max_rank_change": int(
            np.abs(np.argsort(np.argsort(reference)) - np.argsort(np.argsort(exported))).max()
        ),
    }


def _measure(
    run: object, windows: np.ndarray, batch_size: int, iterations: int, warmup: int, name: str
) -> LatencyStats:
    """Time a callable over repeated batches."""
    callable_run = run
    batch = windows[:batch_size]

    cold_started = time.perf_counter()
    callable_run(batch)  # type: ignore[operator]
    cold_ms = (time.perf_counter() - cold_started) * 1000.0

    for _ in range(warmup):
        callable_run(batch)  # type: ignore[operator]

    samples: list[float] = []
    for i in range(iterations):
        offset = (i * batch_size) % max(1, len(windows) - batch_size)
        chunk = windows[offset : offset + batch_size]
        started = time.perf_counter()
        callable_run(chunk)  # type: ignore[operator]
        samples.append((time.perf_counter() - started) * 1000.0)

    array = np.array(samples)
    mean_ms = float(array.mean())
    return LatencyStats(
        runtime=name,
        batch_size=batch_size,
        iterations=iterations,
        p50_ms=float(np.percentile(array, 50)),
        p95_ms=float(np.percentile(array, 95)),
        p99_ms=float(np.percentile(array, 99)),
        mean_ms=mean_ms,
        throughput_per_s=batch_size / (mean_ms / 1000.0) if mean_ms > 0 else float("inf"),
        cold_start_ms=cold_ms,
    )


def benchmark(
    predictor: TemporalPredictor,
    onnx_path: Path,
    windows: np.ndarray,
    batch_sizes: tuple[int, ...] = (1, 32),
    iterations: int = 200,
    warmup: int = 20,
) -> JsonDict:
    """Benchmark both runtimes and report the measured distribution.

    Batch size 1 is the number that matters: live inference scores one window at
    a time, so a throughput figure from a large batch would describe a workload
    the service never runs.

    Args:
        predictor: The PyTorch model.
        onnx_path: The exported artifact.
        windows: Real windows to score.
        batch_sizes: Batch sizes to measure.
        iterations: Timed iterations per configuration.
        warmup: Untimed iterations before measuring.

    Returns:
        A report with per-runtime statistics and the parity check.
    """
    onnx_predictor = OnnxPredictor(onnx_path, predictor.metadata)
    results: list[JsonDict] = []
    for batch_size in batch_sizes:
        for name, run in (
            ("pytorch", predictor.predict_proba),
            ("onnxruntime", onnx_predictor.predict_proba),
        ):
            results.append(_measure(run, windows, batch_size, iterations, warmup, name).to_dict())
    return {
        "device": "cpu",
        "opset": OPSET_VERSION,
        "sequence_length": predictor.metadata.sequence_length,
        "n_features": predictor.metadata.n_features,
        "parity": check_parity(predictor, onnx_path, windows[: min(512, len(windows))]),
        "latency": results,
    }


def write_benchmark(report: JsonDict, path: Path) -> None:
    """Write a benchmark report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
