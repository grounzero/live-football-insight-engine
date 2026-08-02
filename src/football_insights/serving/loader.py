"""Loading the pieces a running service is built from.

Separate from :mod:`football_insights.serving.bootstrap` because the service
itself needs these at *runtime*, not only at startup: switching match rebuilds
the engine and the replay player while the process is up. ``bootstrap`` imports
``create_app`` from ``app``, so ``app`` cannot import ``bootstrap`` back; keeping
the loaders here is what lets both use them.

Loading the best available predictor is deliberately forgiving in one direction
and strict in the other. A missing artifact falls back to the rule-based
predictor, clearly labelled, so the demo runs on a fresh clone before anything
is trained. A *mismatched* artifact does not fall back: a model whose feature
schema disagrees with the running code is refused, readiness goes false, and the
reason is reported. Quietly substituting a different model would be worse than
not starting.

Where a match *comes from* is a :class:`MatchSource`. There are two: the
downloaded Metrica sample data, and a fixture generated in memory. The second
exists so the published container can start with no dataset, no bind mount and
no network — it is the same canonical ``MatchTracking``/``Event``/``Orientation``
the parsers produce, not a parallel schema, so everything downstream is unaware
of the difference. Selection happens once, here, rather than as a mode check
scattered through the route handlers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from football_insights.data.acquire import AVAILABLE_MATCHES, MATCHES_BY_ID, MatchFiles
from football_insights.data.synthetic import DEFAULT_PROFILE, PROFILES, FixtureProfile
from football_insights.errors import DataValidationError, SchemaVersionError
from football_insights.features.spec import DEFAULT_FEATURE_SPEC
from football_insights.models.heuristic import HeuristicPredictor
from football_insights.replay.player import ReplayPlayer
from football_insights.serving.engine import InsightEngine
from football_insights.serving.metrics import Metrics
from football_insights.types import JsonDict

if TYPE_CHECKING:
    from pathlib import Path

    from football_insights.config import Settings
    from football_insights.domain import Event, MatchTracking, Orientation
    from football_insights.models.base import Predictor

LOGGER = logging.getLogger("football_insights.loader")

#: Preference order when no predictor is named in configuration.
PREFERRED = ("gru-temporal", "baseline-gbdt", "baseline-logistic")

#: Identifier prefix for the generated fixtures. Named so nobody can mistake
#: one for a real fixture in a status payload, a log line or the demo's match
#: picker.
SYNTHETIC_MATCH_PREFIX: Final = "Synthetic_Demo"
SYNTHETIC_SOURCE_FORMAT: Final = "synthetic"

#: Retained as the id of the first public fixture, so an existing bookmark,
#: configuration file or `--match Synthetic_Demo` keeps working.
SYNTHETIC_MATCH_ID: Final = SYNTHETIC_MATCH_PREFIX

#: Fixture shape. Five minutes of play at 25 Hz is roughly 75 seconds of
#: wall-clock at the public demo's 4x, so all three archetypes are seen inside
#: four minutes — short enough that a visitor giving the page a couple of
#: minutes still watches the rotation happen rather than one fixture looping.
SYNTHETIC_PERIODS: Final = 1
SYNTHETIC_PERIOD_DURATION_S: Final = 300.0

#: Seed offset for the public fixtures.
#:
#: The hosted rotation must never be a fixture the demo model was fitted,
#: early-stopped or thresholded on, or the page would be showing a model its
#: own training data and calling the result a demonstration. The training seeds
#: live in :mod:`football_insights.models.demo_model` and are derived from a
#: different base; this offset keeps the two sets provably disjoint, and a test
#: asserts it rather than trusting the arithmetic.
PUBLIC_FIXTURE_SEED_BASE: Final = 900_000


def public_fixture_id(profile_key: str) -> str:
    """Match id for one public fixture."""
    return f"{SYNTHETIC_MATCH_PREFIX}_{profile_key}"


def public_fixture_seed(index: int) -> int:
    """Seed for the public fixture at ``index`` in the rotation."""
    return PUBLIC_FIXTURE_SEED_BASE + index


class MatchSource(Protocol):
    """Where one match's tracking, events and orientation come from."""

    @property
    def match_id(self) -> str:
        """Identifier this match is known by."""
        ...

    @property
    def data_source(self) -> str:
        """Coarse provenance, reported by ``/ready`` so the UI can label it."""
        ...

    def load(self) -> tuple[MatchTracking, tuple[Event, ...], Orientation]:
        """Produce the match. Blocking; callers on the event loop use a thread."""
        ...


@dataclass(frozen=True, slots=True)
class MetricaMatchSource:
    """A match parsed from the downloaded Metrica sample data."""

    settings: Settings
    match_id: str

    @property
    def data_source(self) -> str:
        """Real tracking data, downloaded locally and never redistributed."""
        return "metrica"

    def load(self) -> tuple[MatchTracking, tuple[Event, ...], Orientation]:
        """Parse and orient the match.

        Roughly 1.5 seconds, almost all of it parsing two 32 MB tracking files.

        Returns:
            Tracking, events and orientation.

        Raises:
            DataValidationError: If the match is not catalogued, or its files
                have not been downloaded.
        """
        from football_insights.data import metrica_csv, metrica_epts
        from football_insights.data.orientation import infer_orientation

        files = _catalogued(self.match_id)
        paths = files.paths(self.settings.paths.raw_dir)
        _require_downloaded(self.match_id, paths)

        declared = None
        if files.source_format == "metrica_csv":
            tracking, events = metrica_csv.read_match(paths["home"], paths["away"], paths["events"])
        else:
            tracking, events, metadata = metrica_epts.read_match(
                paths["tracking"], paths["metadata"], paths["events"]
            )
            declared = metrica_epts.declared_directions(metadata)

        orientation, home_players, away_players = infer_orientation(
            tracking,
            events,
            self.match_id,
            declared=declared,
            overrides=self.settings.direction_overrides,
            override_reasons=self.settings.direction_override_reasons,
        )
        tracking = type(tracking)(
            period=tracking.period,
            frame=tracking.frame,
            time_s=tracking.time_s,
            home_xy=tracking.home_xy,
            away_xy=tracking.away_xy,
            ball_xy=tracking.ball_xy,
            home_players=home_players,
            away_players=away_players,
            frame_rate=tracking.frame_rate,
        )
        return tracking, events, orientation


@dataclass(frozen=True, slots=True)
class SyntheticDemoMatchSource:
    """A deterministic fixture generated in memory.

    Reads nothing and downloads nothing, which is what makes the published
    container self-contained. Orientation comes from the generator's own ground
    truth rather than being inferred, because here it is known exactly.
    """

    seed: int = 42
    n_periods: int = SYNTHETIC_PERIODS
    period_duration_s: float = SYNTHETIC_PERIOD_DURATION_S
    profile: FixtureProfile = DEFAULT_PROFILE
    match_id: str = SYNTHETIC_MATCH_ID

    @property
    def data_source(self) -> str:
        """Generated, not observed. Surfaced wherever the fixture is named."""
        return SYNTHETIC_SOURCE_FORMAT

    def load(self) -> tuple[MatchTracking, tuple[Event, ...], Orientation]:
        """Generate the fixture.

        Returns:
            Tracking, events and orientation, identical for a given seed.
        """
        from football_insights.data.synthetic import generate_synthetic_match

        match = generate_synthetic_match(
            seed=self.seed,
            n_periods=self.n_periods,
            period_duration_s=self.period_duration_s,
            profile=self.profile,
        )
        return match.tracking, match.events, match.orientation


def demo_fixture_cycle() -> tuple[SyntheticDemoMatchSource, ...]:
    """The public rotation, in order: one fixture per tactical archetype.

    Built here rather than at import so the seeds and the profile list stay in
    one place, and so a caller cannot accidentally rotate through two fixtures
    that differ only by seed.

    Returns:
        One source per profile, each on a seed reserved for public use.
    """
    return tuple(
        SyntheticDemoMatchSource(
            seed=public_fixture_seed(index),
            profile=profile,
            match_id=public_fixture_id(profile.key),
        )
        for index, profile in enumerate(PROFILES)
    )


def resolve_match_source(settings: Settings, match_id: str) -> MatchSource:
    """Choose where a match comes from, by identifier alone.

    Keyed on the id rather than on public-demo mode, so the generated fixture is
    equally available on a development machine that has never downloaded the
    dataset, and so public mode changes only *which match is chosen by default*
    — never the meaning of an explicit request.

    Args:
        settings: Resolved configuration.
        match_id: Match to load.

    Returns:
        The source that can produce it.
    """
    for source in demo_fixture_cycle():
        if source.match_id == match_id:
            return source
    if match_id == SYNTHETIC_MATCH_ID:
        # The bare prefix predates the rotation and is kept working, on the
        # configured seed rather than a public one so a local run with an
        # explicit `replay.seed` still honours it.
        return SyntheticDemoMatchSource(seed=settings.replay.seed)
    return MetricaMatchSource(settings, match_id)


def default_match_id(settings: Settings) -> str:
    """The match to replay when nobody named one."""
    if settings.replay.match_id:
        return settings.replay.match_id
    if settings.service.public_demo:
        return demo_fixture_cycle()[0].match_id
    return "Sample_Game_2"


def _catalogued(match_id: str) -> MatchFiles:
    """Look up a match, failing with the catalogue rather than a bare KeyError.

    Raises:
        DataValidationError: If the id is not in the catalogue.
    """
    try:
        return MATCHES_BY_ID[match_id]
    except KeyError as exc:
        known = ", ".join([*sorted(MATCHES_BY_ID), SYNTHETIC_MATCH_ID])
        msg = f"unknown match {match_id!r}; this build knows: {known}"
        raise DataValidationError(msg) from exc


def _require_downloaded(match_id: str, paths: dict[str, Path]) -> None:
    """Fail before parsing when the dataset is not on disk.

    The service used to discover this several hundred milliseconds into a parse,
    as a ``FileNotFoundError`` naming one CSV. Startup happens before the port is
    bound, so that surfaced as a container that exited with a stack trace and no
    indication that the fix is to run one command.

    Raises:
        DataValidationError: If any required file is missing.
    """
    missing = sorted(str(path) for path in paths.values() if not path.is_file())
    if missing:
        msg = (
            f"match {match_id!r} has not been downloaded: missing {', '.join(missing)}. "
            "Run `football-insights acquire` to fetch the Metrica sample data, or use "
            f"the generated fixture {SYNTHETIC_MATCH_ID!r}, which needs no download."
        )
        raise DataValidationError(msg)


def available_matches(raw_dir: Path, *, public_demo: bool = False) -> tuple[JsonDict, ...]:
    """Every catalogued match, and whether it can actually be replayed.

    Availability is a check against the filesystem rather than a lookup in the
    catalogue. Loading reads the raw tracking files, so a match this build knows
    about but has never downloaded is not playable, and offering it in a selector
    would produce a 1.5-second load ending in a stack trace.

    Args:
        raw_dir: Directory the dataset was downloaded into.
        public_demo: When set, report only the generated fixture. A hosted demo
            has no dataset and cannot switch match, so listing three Metrica
            matches — all of them unavailable — would advertise capability the
            deployment does not have.

    Returns:
        One entry per playable match, in catalogue order.
    """
    if public_demo:
        # All three, and all playable: they are generated, so there is nothing
        # to download and nothing that can be missing. Listing them is not an
        # offer to switch — the mutating routes are still withheld — it is how
        # a viewer can see what the rotation contains.
        return tuple(
            {
                "id": source.match_id,
                "source_format": SYNTHETIC_SOURCE_FORMAT,
                "available": True,
                "name": source.profile.name,
                "narrative": source.profile.narrative,
            }
            for source in demo_fixture_cycle()
        )
    return tuple(
        {
            "id": match.match_id,
            "source_format": match.source_format,
            "available": all(path.is_file() for path in match.paths(raw_dir).values()),
        }
        for match in AVAILABLE_MATCHES
    )


def _load_named(registry: Path, name: str) -> Predictor | None:
    """Load one named artifact in preference order, or ``None`` if absent.

    ``.pt`` is tried first so a development machine with both a checkpoint and
    an export keeps serving the checkpoint, exactly as before. ``.onnx`` is last
    and is what the published container actually uses: it ships no PyTorch, so
    the checkpoint branch is skipped and the exported graph is scored through
    ONNX Runtime.

    Args:
        registry: Directory holding the artifacts.
        name: Artifact base name.

    Returns:
        The predictor, or ``None`` when no artifact of that name exists.
    """
    torch_path = registry / f"{name}.pt"
    if torch_path.is_file():
        try:
            from football_insights.models.temporal import TemporalPredictor
        except ImportError:
            # Not an error: the base install omits torch on purpose, and an
            # export of the same model may sit right beside this checkpoint.
            LOGGER.info(
                "torch is not installed; skipping checkpoint",
                extra={"model": name, "path": str(torch_path)},
            )
        else:
            return TemporalPredictor.load(torch_path)

    pickle_path = registry / f"{name}.pkl"
    if pickle_path.is_file():
        from football_insights.models.baseline import BaselinePredictor

        return BaselinePredictor.load(pickle_path)

    onnx_path = registry / f"{name}.onnx"
    if onnx_path.is_file():
        from football_insights.models.base import ModelMetadata
        from football_insights.models.onnx_predictor import OnnxPredictor

        metadata_path = registry / f"{name}.metadata.json"
        if not metadata_path.is_file():
            # An ONNX graph carries no threshold, no schema hash and no
            # provenance, so serving one without its sidecar would mean
            # inventing all three.
            msg = (
                f"{onnx_path} has no metadata at {metadata_path.name}; refusing to serve a "
                "model whose feature schema and decision threshold are unknown"
            )
            raise DataValidationError(msg)
        metadata = ModelMetadata.read(metadata_path)
        metadata.require_schema(DEFAULT_FEATURE_SPEC.schema_hash)
        return OnnxPredictor(onnx_path, metadata)

    return None


def load_predictor(settings: Settings) -> Predictor:
    """Load the best available predictor, falling back to the heuristic.

    Args:
        settings: Resolved configuration.

    Returns:
        A predictor. The rule-based fallback is returned when no trained
        artifact is available, with ``is_ml`` false so nothing downstream can
        present it as a model.

    Raises:
        SchemaVersionError: If an artifact was built against a different
            feature schema.
    """
    registry = settings.paths.registry_dir
    names: list[str] = (
        [settings.model.model_name]
        if settings.model.model_name
        else ([] if settings.model.predictor == "heuristic" else list(PREFERRED))
    )

    for name in names:
        try:
            predictor = _load_named(registry, name)
        except SchemaVersionError:
            # Do not fall through to another model: a schema mismatch means the
            # feature code has changed under a trained artifact, and the right
            # response is to report it, not to silently serve something else.
            LOGGER.exception("refusing model with mismatched feature schema", extra={"model": name})
            raise
        if predictor is None:
            continue
        LOGGER.info("loaded model", extra={"model": name, "is_ml": predictor.metadata.is_ml})
        return predictor

    LOGGER.warning(
        "no trained artifact found; using the rule-based fallback",
        extra={"registry": str(registry), "is_ml": False},
    )
    return HeuristicPredictor(settings.model.threshold)


def load_match(
    settings: Settings, match_id: str
) -> tuple[MatchTracking, tuple[Event, ...], Orientation]:
    """Load one match for replay, from wherever it comes from.

    Blocking and, for Metrica data, not cheap — roughly 1.5 seconds per match,
    almost all of it parsing two 32 MB tracking files. Callers on the event loop
    must run this in a worker thread.

    Args:
        settings: Resolved configuration.
        match_id: Which match to load.

    Returns:
        Tracking, events and orientation.
    """
    return resolve_match_source(settings, match_id).load()


def build_engine(
    settings: Settings,
    tracking: MatchTracking,
    events: tuple[Event, ...],
    orientation: Orientation,
    predictor: Predictor | None,
    metrics: Metrics,
) -> InsightEngine:
    """Construct the inference engine for a loaded match."""
    threshold = predictor.metadata.threshold if predictor is not None else settings.model.threshold
    tuned = settings.model_copy(deep=True)
    tuned.model.threshold = threshold
    return InsightEngine(
        settings=tuned,
        orientation=orientation,
        events=events,
        frame_rate=tracking.frame_rate,
        predictor=predictor,
        metrics=metrics,
        home_is_gk=np.array([p.is_goalkeeper for p in tracking.home_players]),
        away_is_gk=np.array([p.is_goalkeeper for p in tracking.away_players]),
        spec=DEFAULT_FEATURE_SPEC,
    )


def load_replay(
    settings: Settings,
    match_id: str,
    fault_profile: str,
    seed: int,
    speed: float = 0.0,
) -> tuple[InsightEngine, ReplayPlayer]:
    """Build an engine and a replay player for one match."""
    tracking, events, orientation = load_match(settings, match_id)
    metrics = Metrics()
    predictor = load_predictor(settings)
    engine = build_engine(settings, tracking, events, orientation, predictor, metrics)
    player = ReplayPlayer(
        match_id=match_id,
        tracking=tracking,
        profile=settings.fault_profile(fault_profile),
        seed=seed,
        speed=speed,
    )
    return engine, player


def rebuild_for_match(
    settings: Settings,
    match_id: str,
    predictor: Predictor | None,
    metrics: Metrics,
    fault_profile: str,
    seed: int,
    speed: float,
) -> tuple[InsightEngine, ReplayPlayer]:
    """Build an engine and player for a *different* match on a running service.

    Separate from :func:`load_replay` because the two differ in what they are
    allowed to create. A running service already has a predictor and a metric
    registry, and both must survive the swap: the predictor is match-independent
    so reloading it would cost seconds for nothing, and a second ``Metrics``
    would build a registry that ``GET /metrics`` never reads, leaving every
    counter apparently frozen from the moment someone changed match.

    Args:
        settings: Resolved configuration.
        match_id: Match to switch to.
        predictor: The predictor already in use.
        metrics: The metric registry already being served.
        fault_profile: Fault profile name to carry over.
        seed: Fault-injection seed to carry over.
        speed: Replay rate to carry over.

    Returns:
        The new engine and player. Neither is installed; the caller owns that.
    """
    tracking, events, orientation = load_match(settings, match_id)
    engine = build_engine(settings, tracking, events, orientation, predictor, metrics)
    player = ReplayPlayer(
        match_id=match_id,
        tracking=tracking,
        profile=settings.fault_profile(fault_profile),
        seed=seed,
        speed=speed,
    )
    return engine, player
