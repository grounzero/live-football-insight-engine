"""Typed configuration.

Every tunable lives here. Values come from three layers: model defaults,
environment variables prefixed ``FI_`` with ``__`` as the nesting delimiter
(``FI_WINDOW__HORIZON_S=7``), and an optional YAML file.

Note the precedence carefully, because it is not the obvious one: a YAML file is
passed to the model as init keyword arguments, and pydantic-settings ranks init
arguments *above* environment variables. So a key present in ``--config`` wins
over the matching ``FI_`` variable. That is long-standing behaviour and every
CLI command depends on it, so it is documented rather than changed here.

The bind address is the one place that would be actively dangerous to leave to
that ordering — a hosting platform injects ``PORT`` and expects to be obeyed —
so :func:`resolve_port` and :func:`resolve_host` sit above all three layers and
are the only place in the package that reads those variables.

The episode-grouping and threshold knobs are load-bearing for reported results,
so :meth:`Settings.fingerprint` hashes the whole resolved configuration into
model metadata and evaluation reports.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from football_insights.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping

PredictorKind = Literal["heuristic", "logistic", "gbdt", "gru"]
InferenceBackend = Literal["torch", "onnx"]

#: Platform-injected listen port. Railway, Fly, Heroku and Cloud Run all set
#: this and route to whatever it names; a service that ignores it is
#: unreachable no matter how healthy it is.
PORT_ENV_VAR: Final = "PORT"

MIN_PORT: Final = 1
MAX_PORT: Final = 65535


def _parse_yaml_mapping(text: str, source: Path) -> dict[str, Any]:
    """Parse a YAML document into a string-keyed mapping.

    This is where the ``Any`` that ``yaml.safe_load`` returns stops: everything
    downstream of it is a validated Pydantic model. YAML also permits non-string
    mapping keys (``1: x``, ``true: y``), which would otherwise surface much
    later as an unreadable ``TypeError`` from ``Settings(**data)``.

    Args:
        text: Raw YAML source.
        source: Path the text came from, used in error messages.

    Returns:
        The document as a string-keyed mapping; empty for an empty document.

    Raises:
        TypeError: If the document is not a mapping, or has a non-string key.
    """
    document: object = yaml.safe_load(text)
    if document is None:
        return {}
    if not isinstance(document, dict):
        msg = f"configuration file {source} must contain a mapping"
        raise TypeError(msg)
    # isinstance() cannot recover the element types of a dict[Unknown, Unknown];
    # the cast states what a YAML mapping actually is, and the loop below is the
    # check that makes it true rather than assumed.
    entries = cast("dict[object, object]", document)
    parsed: dict[str, Any] = {}
    for key, value in entries.items():
        if not isinstance(key, str):
            msg = f"configuration file {source}: mapping keys must be strings, got {key!r}"
            raise TypeError(msg)
        parsed[key] = value
    return parsed


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
    #: Expose the pipeline stages (acquire, prepare, train, evaluate, benchmark)
    #: as job endpoints, and the panel that drives them.
    #:
    #: Off by default, and the routes are not registered at all when it is off.
    #: The service has no authentication and mounts the demo at ``/``, so anyone
    #: who can reach the port could otherwise start a 180 MB download and pin the
    #: CPU for minutes. That is fine on a development machine and is not fine in
    #: the published container, so turning it on is a deliberate act:
    #: ``serve --dev-tools`` or ``FI_SERVICE__ENABLE_PIPELINE_CONTROLS=1``.
    enable_pipeline_controls: bool = False
    #: Run as a hosted public demo: a generated synthetic fixture instead of
    #: Metrica data, and a read-only replay.
    #:
    #: Replay state is process-wide — one visitor pausing the match pauses it
    #: for everyone watching — so on a public URL the mutating replay routes are
    #: not registered at all, exactly as ``enable_pipeline_controls`` withholds
    #: the job routes. Off by default: turning it on is a deliberate act, via
    #: ``serve --public-demo`` or ``FI_SERVICE__PUBLIC_DEMO=1``.
    public_demo: bool = False


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
            data = _parse_yaml_mapping(config_path.read_text(), config_path)
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
        return cls(**data)


def default_settings() -> Settings:
    """Settings with every default in place, used by tests and the demo."""
    return Settings()


def resolve_port(
    explicit: int | None,
    settings: Settings,
    env: Mapping[str, str] | None = None,
) -> int:
    """Decide which port to listen on.

    Precedence, highest first:

    1. ``explicit`` — the ``--port`` option, an operator saying so directly;
    2. ``PORT`` — the hosting platform, which routes to this and nothing else;
    3. ``settings.service.port`` — ``FI_SERVICE__PORT`` or a YAML file;
    4. the model default, 8000.

    ``PORT`` sits above the ``FI_``-prefixed layer deliberately. A platform that
    injects it will send traffic there regardless of what the image was built
    with, so honouring a baked-in ``FI_SERVICE__PORT`` instead would produce a
    container that looks healthy and is unreachable.

    Args:
        explicit: Value passed on the command line, if any.
        settings: Resolved configuration.
        env: Environment to read; the real one when omitted.

    Returns:
        A port between 1 and 65535.

    Raises:
        ConfigurationError: If ``explicit`` or ``PORT`` is out of range, or
            ``PORT`` is not an integer.
    """
    if explicit is not None:
        return _validated_port(explicit, source="--port")

    raw = (env if env is not None else os.environ).get(PORT_ENV_VAR)
    if raw is not None and raw.strip():
        try:
            parsed = int(raw.strip())
        except ValueError as exc:
            msg = (
                f"{PORT_ENV_VAR}={raw!r} is not an integer. It is set by the hosting "
                "platform and must name a TCP port."
            )
            raise ConfigurationError(msg) from exc
        return _validated_port(parsed, source=PORT_ENV_VAR)

    # Already validated by the ge/le bounds on ServiceSettings.port.
    return settings.service.port


def _validated_port(value: int, source: str) -> int:
    """Bounds-check a port, naming where it came from."""
    if not MIN_PORT <= value <= MAX_PORT:
        msg = f"{source}={value} is outside the valid port range {MIN_PORT}-{MAX_PORT}"
        raise ConfigurationError(msg)
    return value


def resolve_host(explicit: str | None, settings: Settings) -> str:
    """Decide which interface to bind.

    ``--host`` wins, then ``settings.service.host`` — which the container image
    sets to ``0.0.0.0`` through ``FI_SERVICE__HOST``, because the loopback
    default that is right on a development machine makes a container
    unreachable from outside itself.

    Args:
        explicit: Value passed on the command line, if any.
        settings: Resolved configuration.

    Returns:
        The interface to bind.
    """
    return explicit if explicit is not None else settings.service.host
