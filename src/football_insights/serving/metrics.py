"""Prometheus metrics.

Two namespaces, kept strictly apart because they answer different questions:

``fi_model_*``
    How well the predictor is working — throughput, latency, confidence spread,
    how often it was handed an unusable window.

``fi_insight_*``
    What the audience actually experienced — how many candidates were produced,
    how many became insights, and why the rest did not.

Conflating them would hide the most operationally useful signal in the system:
a model behaving normally while the editorial layer suppresses everything is a
completely different incident from a model that has stopped firing.

Every metric here is genuinely incremented on the live path. Nothing is declared
for appearance.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

#: Latency buckets in seconds. Resolution is concentrated between 1 ms and
#: 100 ms, which is where the inference and end-to-end budgets sit.
LATENCY_BUCKETS: Final = (
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)

CONFIDENCE_BUCKETS: Final = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


class Metrics:
    """Metric collection bound to one registry.

    A dedicated registry rather than the global default keeps tests isolated:
    each constructs its own instance and asserts on its own counters.
    """

    __slots__ = (
        "_registry",
        "candidates",
        "confidence",
        "drift_violations",
        "e2e_latency",
        "emitted",
        "feature_validation_failures",
        "frames",
        "inference_latency",
        "invalid_windows",
        "missing_frames",
        "model_info",
        "predictions",
        "ready",
        "replay_frames",
        "requests",
        "suppressed",
    )

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Create the metric families.

        Args:
            registry: Registry to bind to; a fresh one is created when omitted.
        """
        self._registry = registry if registry is not None else CollectorRegistry()
        r = self._registry

        # ------------------------------------------------------ model stage
        self.predictions = Counter(
            "fi_model_predictions_total",
            "Windows scored by the predictor.",
            ["model", "is_ml"],
            registry=r,
        )
        self.inference_latency = Histogram(
            "fi_model_inference_latency_seconds",
            "Time spent inside the predictor.",
            ["model", "backend"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.confidence = Histogram(
            "fi_model_confidence",
            "Distribution of predicted probabilities.",
            ["model"],
            buckets=CONFIDENCE_BUCKETS,
            registry=r,
        )
        self.invalid_windows = Counter(
            "fi_model_invalid_window_total",
            "Windows rejected before scoring.",
            ["reason"],
            registry=r,
        )
        self.model_info = Gauge(
            "fi_model_info",
            "Loaded model metadata; value is 1 while that model is active.",
            ["model", "version", "kind", "is_ml", "feature_schema"],
            registry=r,
        )

        # -------------------------------------------------- editorial stage
        self.candidates = Counter(
            "fi_insight_candidates_total",
            "Predictions that cleared the decision threshold.",
            ["kind"],
            registry=r,
        )
        self.emitted = Counter(
            "fi_insight_emitted_total",
            "Insights shown to viewers.",
            ["kind", "is_ml"],
            registry=r,
        )
        self.suppressed = Counter(
            "fi_insight_suppressed_total",
            "Candidates withheld, by reason.",
            ["reason"],
            registry=r,
        )

        # -------------------------------------------------------- pipeline
        self.e2e_latency = Histogram(
            "fi_pipeline_end_to_end_latency_seconds",
            "Frame arrival to insight decision.",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.frames = Counter(
            "fi_pipeline_frames_total",
            "Frames accepted into the rolling window.",
            registry=r,
        )
        self.missing_frames = Counter(
            "fi_pipeline_missing_frames_total",
            "Gaps detected in the incoming frame sequence.",
            registry=r,
        )
        self.replay_frames = Counter(
            "fi_replay_frames_total",
            "Frames handled by the replay layer, by outcome.",
            ["outcome"],
            registry=r,
        )
        self.feature_validation_failures = Counter(
            "fi_feature_validation_failures_total",
            "Feature windows that failed validation.",
            ["check"],
            registry=r,
        )
        self.drift_violations = Counter(
            "fi_drift_violations_total",
            "Monitoring checks that breached their threshold.",
            ["check"],
            registry=r,
        )

        # --------------------------------------------------------- service
        self.requests = Counter(
            "fi_service_requests_total",
            "HTTP requests handled.",
            ["endpoint", "method", "status"],
            registry=r,
        )
        self.ready = Gauge(
            "fi_service_ready",
            "1 when a predictor is loaded and the service can serve predictions.",
            registry=r,
        )
        self.ready.set(0)

    @property
    def registry(self) -> CollectorRegistry:
        """The underlying registry."""
        return self._registry

    def set_model(self, name: str, version: str, kind: str, is_ml: bool, schema: str) -> None:
        """Publish the active model as an info gauge."""
        self.model_info.clear()
        self.model_info.labels(
            model=name,
            version=version,
            kind=kind,
            is_ml=str(is_ml).lower(),
            feature_schema=schema,
        ).set(1)

    def render(self) -> tuple[bytes, str]:
        """Render the registry for the ``/metrics`` endpoint.

        Returns:
            The exposition payload and its content type.
        """
        return generate_latest(self._registry), CONTENT_TYPE_LATEST

    def snapshot(self) -> dict[str, float]:
        """Flat sample map, used by tests and the demo status strip.

        Returns:
            Mapping of fully qualified sample name to value. Labelled samples
            are keyed ``name{label="value",...}``, matching the exposition
            format so a key can be copied straight from ``/metrics``.
        """
        out: dict[str, float] = {}
        for metric in self._registry.collect():
            for sample in metric.samples:
                key = sample.name
                if sample.labels:
                    labels = ",".join(f'{k}="{v}"' for k, v in sorted(sample.labels.items()))
                    key = f"{key}{{{labels}}}"
                out[key] = sample.value
        return out
