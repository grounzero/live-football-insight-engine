"""Typed configuration.

Every tunable lives here. Values resolve in three layers, later overriding
earlier: dataclass defaults, an optional YAML file, then environment variables
prefixed ``FI_`` with ``__`` as the nesting delimiter (``FI_WINDOW__HORIZON_S=7``).

The episode-grouping and threshold knobs are load-bearing for reported results,
so :meth:`Settings.fingerprint` hashes the whole resolved configuration into
model metadata and evaluation reports.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PredictorKind = Literal["heuristic", "logistic", "gbdt", "gru"]
InferenceBackend = Literal["torch", "onnx"]


class PathSettings(BaseModel):
    """Filesystem locations. All relative paths resolve against the repo root."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    registry_dir: Path = Path("artifacts/registry")
    reports_dir: Path = Path("artifacts/reports")
    mlflow_uri: str = "file:./artifacts/mlruns"


class WindowSettings(BaseModel):
    """Observation window, prediction horizon and sampling.

    ``observation_s`` of history is used to predict whether the target occurs
    within the next ``horizon_s``. Samples are emitted every ``stride_s``.
    """

    observation_s: float = Field(default=5.0, gt=0)
    horizon_s: float = Field(default=5.0, gt=0)
    stride_s: float = Field(default=0.5, gt=0)
    #: Frames per second the window is resampled to before it reaches a model.
    #: 25 Hz raw tracking at 10 Hz gives 50 timesteps for a 5 s window.
    sample_hz: float = Field(default=10.0, gt=0)
    #: Fraction of expected frames that must be present and finite for a window
    #: to be scored at all. Below this the window is structurally invalid and
    #: no insight may be emitted from it.
    min_valid_frame_ratio: float = Field(default=0.8, ge=0.0, le=1.0)

    @property
    def sequence_length(self) -> int:
        """Number of timesteps a model sees per window."""
        return round(self.observation_s * self.sample_hz)


class EpisodeSettings(BaseModel):
    """Episode grouping for evaluation.

    These knobs materially move episode-level precision, so they are frozen
    from the training matches before the held-out match is scored, recorded in
    the run config, and reported with a sensitivity grid.
    """

    #: Penalty-area entries by the same team closer together than this collapse
    #: into a single ground-truth episode.
    merge_gap_s: float = Field(default=10.0, gt=0)
    #: Gaps up to this length are bridged when grouping consecutive firing
    #: windows into one alarm, absorbing probability flicker around the threshold.
    alarm_bridge_gap_s: float = Field(default=2.0, ge=0)
    #: Resampling unit count for the cluster bootstrap.
    bootstrap_replicates: int = Field(default=2000, ge=100)
    bootstrap_seed: int = 20260801
    #: Operating points are chosen to respect this interruption budget rather
    #: than to maximise a window-level score. Measured on the training matches,
    #: a window-precision target of 0.30 produced ~140 false alarms per 90
    #: minutes, which no viewer-facing product could use.
    max_false_alarms_per_90: float = Field(default=12.0, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)


class ModelSettings(BaseModel):
    """Model selection, thresholds and runtime backend."""

    #: Prefer the trained temporal model. When no artifact is present — a fresh
    #: clone, or before `make train` — loading falls back to the rule-based
    #: predictor, clearly labelled. Set to "heuristic" to force the fallback.
    predictor: PredictorKind = "gru"
    backend: InferenceBackend = "torch"
    #: Probability at or above which a prediction becomes an insight candidate.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_name: str | None = None
    seed: int = 20260801

    # Temporal model hyperparameters.
    hidden_size: int = Field(default=48, ge=1)
    num_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=256, ge=1)
    max_epochs: int = Field(default=60, ge=1)
    early_stopping_patience: int = Field(default=8, ge=1)
    #: Weight applied to the positive class to counter roughly 20:1 imbalance.
    positive_class_weight: float | None = None


class EditorialSettings(BaseModel):
    """Editorial relevance and suppression.

    This stage is deliberately separate from the model: a statistically valid
    prediction does not automatically become a viewer-facing insight.
    """

    #: Minimum gap between two emitted insights of the same kind.
    cooldown_s: float = Field(default=20.0, ge=0)
    #: A candidate older than this is dropped rather than shown late.
    max_staleness_s: float = Field(default=2.0, ge=0)
    #: A candidate whose wording matches a recent one is treated as duplicate.
    duplicate_window_s: float = Field(default=45.0, ge=0)
    #: Consecutive windows above threshold required before the first emission,
    #: which suppresses single-frame probability spikes.
    min_consecutive_windows: int = Field(default=2, ge=1)
    #: Never emit when the ball is already inside the penalty area: there is
    #: nothing left to predict.
    suppress_when_in_box: bool = True


class FaultProfileSettings(BaseModel):
    """One named replay degradation profile.

    Every field is a probability or a range consumed in a fixed draw order, so
    the emitted stream is a pure function of ``(profile, seed, source frames)``.
    """

    name: str = "clean"
    jitter_ms: tuple[float, float] = (0.0, 0.0)
    drop_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    delay_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    delay_ms: tuple[float, float] = (0.0, 0.0)
    reorder_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    reorder_window: int = Field(default=0, ge=0)


class ReplaySettings(BaseModel):
    """Replay pacing and fault injection."""

    #: 1.0 is real time; higher is faster. 0 means emit as fast as possible.
    speed: float = Field(default=1.0, ge=0.0)
    fault_profile: str = "clean"
    seed: int = 42
    match_id: str | None = None
    loop: bool = False


class ServiceSettings(BaseModel):
    """HTTP service."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    #: Emitted in structured logs and echoed on responses for correlation.
    service_name: str = "football-insights"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)


class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="FI_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    paths: PathSettings = Field(default_factory=PathSettings)
    window: WindowSettings = Field(default_factory=WindowSettings)
    episode: EpisodeSettings = Field(default_factory=EpisodeSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    editorial: EditorialSettings = Field(default_factory=EditorialSettings)
    replay: ReplaySettings = Field(default_factory=ReplaySettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    fault_profiles: dict[str, FaultProfileSettings] = Field(default_factory=dict)
    #: Per match and period manual orientation overrides, keyed
    #: ``"<match_id>:<period>:<team>"`` with a value of ``"+x"`` or ``"-x"``.
    #: Each override requires a reason, recorded in the direction report.
    direction_overrides: dict[str, str] = Field(default_factory=dict)
    direction_override_reasons: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_default_profiles(self) -> Settings:
        """Guarantee the four documented fault profiles always exist."""
        defaults = {
            "clean": FaultProfileSettings(name="clean"),
            "jitter": FaultProfileSettings(name="jitter", jitter_ms=(0.0, 25.0)),
            "degraded": FaultProfileSettings(
                name="degraded",
                jitter_ms=(0.0, 40.0),
                drop_prob=0.02,
                duplicate_prob=0.01,
                delay_prob=0.03,
                delay_ms=(40.0, 200.0),
                reorder_prob=0.01,
                reorder_window=3,
            ),
            "hostile": FaultProfileSettings(
                name="hostile",
                jitter_ms=(0.0, 120.0),
                drop_prob=0.10,
                duplicate_prob=0.04,
                delay_prob=0.10,
                delay_ms=(80.0, 600.0),
                reorder_prob=0.05,
                reorder_window=8,
            ),
        }
        for key, profile in defaults.items():
            self.fault_profiles.setdefault(key, profile)
        return self

    @model_validator(mode="after")
    def _validate_overrides(self) -> Settings:
        """Every orientation override must carry a reason."""
        missing = set(self.direction_overrides) - set(self.direction_override_reasons)
        if missing:
            msg = (
                "direction_overrides without a matching direction_override_reasons entry: "
                f"{sorted(missing)}. Overriding inferred playing direction requires a "
                "written justification; it is recorded in the direction report."
            )
            raise ValueError(msg)
        return self

    def fault_profile(self, name: str | None = None) -> FaultProfileSettings:
        """Look up a fault profile by name, defaulting to the replay setting."""
        key = name or self.replay.fault_profile
        try:
            return self.fault_profiles[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.fault_profiles))
            msg = f"unknown fault profile {key!r}; known profiles: {known}"
            raise KeyError(msg) from exc

    def fingerprint(self) -> str:
        """Stable short hash of the resolved configuration.

        Recorded alongside model artifacts and evaluation reports so a set of
        results can always be traced back to the settings that produced it.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def load(cls, path: Path | str | None = None, **overrides: Any) -> Settings:
        """Load settings from an optional YAML file plus environment variables.

        Args:
            path: YAML configuration file. Missing files raise, so a typo in a
                ``--config`` argument never silently falls back to defaults.
            **overrides: Values applied on top of the file, used by the CLI.

        Returns:
            The resolved settings.
        """
        data: dict[str, Any] = {}
        if path is not None:
            config_path = Path(path)
            if not config_path.is_file():
                msg = f"configuration file not found: {config_path}"
                raise FileNotFoundError(msg)
            loaded = yaml.safe_load(config_path.read_text()) or {}
            if not isinstance(loaded, dict):
                msg = f"configuration file {config_path} must contain a mapping"
                raise TypeError(msg)
            data = loaded
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
        return cls(**data)


def default_settings() -> Settings:
    """Settings with every default in place, used by tests and the demo."""
    return Settings()
