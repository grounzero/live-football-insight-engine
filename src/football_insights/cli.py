"""Command-line interface.

Every stage of the pipeline is reachable from here, and every command takes the
same ``--config`` file, so a reviewer can reproduce the reference run without
reading any Python.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from football_insights.config import Settings
from football_insights.errors import ConfigurationError
from football_insights.serving.logging import configure_logging

app = typer.Typer(
    name="football-insights",
    help="Live football insight engine: tracking data to qualified viewer insights.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML configuration file.")
]
MatchOption = Annotated[
    list[str] | None, typer.Option("--match", "-m", help="Match id; repeatable.")
]


def _settings(config: Path | None) -> Settings:
    """Resolve settings and configure logging."""
    settings = Settings.load(config)
    configure_logging(settings.service.log_level)
    return settings


def _echo_json(payload: object) -> None:
    """Print a JSON payload."""
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.command()
def acquire(
    config: ConfigOption = None,
    match: MatchOption = None,
    force: Annotated[bool, typer.Option(help="Re-download even if present.")] = False,
) -> None:
    """Download the Metrica sample data and write a checksummed manifest."""
    from football_insights.data.acquire import acquire as run

    settings = _settings(config)
    manifest = run(settings.paths.raw_dir, match, force)
    typer.echo(f"dataset fingerprint: {manifest['fingerprint']}")
    for entry in manifest["matches"]:
        size = sum(f["bytes"] for f in entry["files"].values())
        typer.echo(f"  {entry['match_id']:16s} {entry['source_format']:14s} {size / 1e6:7.1f} MB")


@app.command()
def prepare(config: ConfigOption = None, match: MatchOption = None) -> None:
    """Parse, validate, orient, feature-ise and label the matches."""
    from football_insights.data.pipeline import prepare_dataset

    settings = _settings(config)
    report = prepare_dataset(settings, match)
    for entry in report["matches"]:
        labels = entry["labels"]
        typer.echo(
            f"  {labels['match_id']:16s} samples={labels['samples']:6d} "
            f"positives={labels['positives']:5d} ({labels['positive_rate']:.2%}) "
            f"episodes={labels['episodes']:3d}"
        )
    typer.echo(f"reports written to {settings.paths.reports_dir}")


@app.command()
def train(
    config: ConfigOption = None,
    test_match: Annotated[str, typer.Option(help="Match held out for reporting.")] = (
        "Sample_Game_2"
    ),
    match: MatchOption = None,
) -> None:
    """Train the reference models and register their artifacts."""
    from football_insights.data.acquire import load_manifest
    from football_insights.models.train import train_reference

    settings = _settings(config)
    fingerprint = load_manifest(settings.paths.raw_dir).get("fingerprint")
    report = train_reference(settings, test_match, match, str(fingerprint))
    for name, result in report["results"]["models"].items():
        window = result["window"]
        episode = result["episode"]
        typer.echo(
            f"  {name:20s} PR-AUC={window['pr_auc']:.3f} "
            f"episode P={episode['precision']:.3f} R={episode['recall']:.3f} "
            f"FA/90={episode['false_alarms_per_90']:.1f}"
        )
    typer.echo(f"artifacts in {settings.paths.registry_dir}")


@app.command()
def evaluate(
    config: ConfigOption = None,
    match: MatchOption = None,
    bootstrap: Annotated[bool, typer.Option(help="Compute bootstrap intervals.")] = True,
) -> None:
    """Run leave-one-match-out cross-validation."""
    from football_insights.data.acquire import load_manifest
    from football_insights.models.train import (
        run_cross_validation,
        write_cross_validation_report,
    )

    settings = _settings(config)
    fingerprint = load_manifest(settings.paths.raw_dir).get("fingerprint")
    report = run_cross_validation(settings, match, str(fingerprint), bootstrap)
    for name, aggregate in report["aggregate"].items():
        typer.echo(
            f"  {name:20s} PR-AUC per fold {aggregate['window_pr_auc']} "
            f"episode P={aggregate['episode_precision_mean']:.3f} "
            f"R={aggregate['episode_recall_mean']:.3f}"
        )
    path = settings.paths.reports_dir / "cross_validation.json"
    write_cross_validation_report(report, path)
    typer.echo(f"report written to {path}")


@app.command()
def export(config: ConfigOption = None) -> None:
    """Export the temporal model to ONNX and check runtime parity."""
    from football_insights.models.export_onnx import check_parity
    from football_insights.models.export_onnx import export as run
    from football_insights.models.temporal import TemporalPredictor
    from football_insights.models.train import load_matches

    settings = _settings(config)
    predictor = TemporalPredictor.load(settings.paths.registry_dir / "gru-temporal.pt")
    path = run(predictor, settings.paths.registry_dir / "gru-temporal.onnx")
    windows = load_matches(settings)[0].windows[:512]
    _echo_json(check_parity(predictor, path, windows))


@app.command()
def demo_model(
    config: ConfigOption = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Registry directory to write into; the configured one by default."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Base seed for fixtures and training.")] = 20260801,
) -> None:
    """Train and export the synthetic-data model the public demo serves.

    Needs the ``train`` extra. Run during the container build, not at startup:
    the artifact is deterministic for a seed, so building it once per image is
    both cheaper and more reproducible than training on every boot.
    """
    from football_insights.models.demo_model import build_demo_model

    settings = _settings(config)
    _echo_json(build_demo_model(settings, out or settings.paths.registry_dir, seed=seed))


@app.command()
def benchmark(
    config: ConfigOption = None,
    iterations: Annotated[int, typer.Option(help="Timed iterations per configuration.")] = 300,
) -> None:
    """Benchmark PyTorch against ONNX Runtime and write the report."""
    from football_insights.models.export_onnx import benchmark as run
    from football_insights.models.export_onnx import export, write_benchmark
    from football_insights.models.temporal import TemporalPredictor
    from football_insights.models.train import load_matches

    settings = _settings(config)
    predictor = TemporalPredictor.load(settings.paths.registry_dir / "gru-temporal.pt")
    # Always re-export. Reusing an existing file benchmarks whatever was
    # exported last, which after a retrain is a different model — the parity
    # check catches it, but only after reporting latency for the wrong graph.
    path = export(predictor, settings.paths.registry_dir / "gru-temporal.onnx")
    windows = load_matches(settings)[0].windows[:2000]
    report = run(predictor, path, windows, iterations=iterations)
    write_benchmark(report, settings.paths.reports_dir / "benchmark.json")
    for row in report["latency"]:
        typer.echo(
            f"  {row['runtime']:12s} batch={row['batch_size']:3d} "
            f"p50={row['p50_ms']:.3f}ms p95={row['p95_ms']:.3f}ms p99={row['p99_ms']:.3f}ms"
        )
    typer.echo(f"  parity max|diff| = {report['parity']['max_abs_diff']:.2e}")


@app.command()
def drift(
    config: ConfigOption = None, reference: str = "Sample_Game_1", target: str = "Sample_Game_2"
) -> None:
    """Compare two matches and report data-quality and distribution drift."""
    from football_insights.monitoring.drift import drift_report, write_drift_report

    settings = _settings(config)
    report = drift_report(settings, reference, target)
    path = settings.paths.reports_dir / "drift.json"
    write_drift_report(report, path)
    typer.echo(f"checks run: {report['checks_run']}, violations: {report['violations']}")
    for check in report["details"]:
        if check["violation"]:
            typer.echo(f"  ! {check['name']}: {check['detail']}")
    typer.echo(f"report written to {path}")


@app.command()
def replay(
    config: ConfigOption = None,
    match: Annotated[str, typer.Option(help="Match to replay.")] = "Sample_Game_2",
    fault_profile: Annotated[str, typer.Option(help="clean, jitter, degraded or hostile.")] = (
        "clean"
    ),
    seed: Annotated[int, typer.Option(help="Fault-injection seed.")] = 42,
    limit: Annotated[int, typer.Option(help="Frames to process; 0 for all.")] = 0,
) -> None:
    """Replay a match through the engine offline and report what happened."""
    from football_insights.serving.bootstrap import load_replay

    settings = _settings(config)
    logging.disable(logging.INFO)
    engine, player = load_replay(settings, match, fault_profile, seed)
    model_name = engine.predictor.metadata.name if engine.predictor else "none"
    typer.echo(
        f"replaying {match} profile={fault_profile} seed={seed} "
        f"frames={player.total_frames} predictor={model_name}"
    )
    emitted = 0
    for index, item in enumerate(player.emitted):
        if limit and index >= limit:
            break
        result = engine.process(item.frame)
        if result.outcome is not None and result.outcome.insight is not None:
            insight = result.outcome.insight
            emitted += 1
            typer.echo(
                f"  [{insight.match_time_s:7.1f}s] {insight.headline}: {insight.detail} "
                f"(p={insight.probability:.2f})"
            )
    _echo_json({"insights": emitted, "fault_summary": player.summary.to_dict()})


@app.command()
def serve(
    config: ConfigOption = None,
    match: Annotated[
        str | None,
        typer.Option(help="Match to replay; the generated fixture in public-demo mode."),
    ] = None,
    fault_profile: Annotated[str, typer.Option()] = "clean",
    seed: Annotated[int, typer.Option()] = 42,
    speed: Annotated[
        float | None,
        typer.Option(
            help=(
                "Replay rate; 1.0 is real time. Defaults to 4x in public-demo mode "
                "and 8x otherwise, unless replay.speed is configured."
            )
        ),
    ] = None,
    host: Annotated[str | None, typer.Option(help="Interface to bind.")] = None,
    port: Annotated[
        int | None,
        typer.Option(help="Port to bind; overrides the PORT environment variable."),
    ] = None,
    dev_tools: Annotated[
        bool,
        typer.Option(help="Expose the pipeline stages as job endpoints and a panel in the demo."),
    ] = False,
    public_demo: Annotated[
        bool,
        typer.Option(help="Serve a generated fixture with a read-only, looping replay."),
    ] = False,
) -> None:
    """Start the API and the viewer demo.

    Raises:
        ConfigurationError: If the requested combination of flags cannot be
            served, or the bind port is invalid.
    """
    import uvicorn

    from football_insights.config import resolve_host, resolve_port, resolve_replay_speed
    from football_insights.serving.bootstrap import create_configured_app

    settings = _settings(config)
    if dev_tools and public_demo:
        # Refused rather than resolved by precedence. The two flags express
        # opposite intentions — one opens the pipeline to anyone who can reach
        # the port, the other assumes anyone can — and silently honouring
        # either one would be a security decision made by argument order.
        msg = (
            "--dev-tools and --public-demo cannot be combined: pipeline controls start "
            "long, resource-hungry work and this service has no authentication"
        )
        raise ConfigurationError(msg)
    if dev_tools:
        # Set here rather than defaulted on, so an operator who did not ask for
        # the pipeline routes never gets them. See ServiceSettings for why.
        settings.service.enable_pipeline_controls = True
    if public_demo:
        settings.service.public_demo = True

    # After the flags above have been folded into `settings`, because the public
    # demo's default rate is one of the things they decide.
    application = create_configured_app(
        settings, match, fault_profile, seed, resolve_replay_speed(speed, settings)
    )
    uvicorn.run(
        application,
        host=resolve_host(host, settings),
        port=resolve_port(port, settings),
        log_level=settings.service.log_level.lower(),
        # One worker, stated rather than implied. The replay, its subscriber set
        # and the Prometheus registry all live in this process, so a second
        # worker would serve a second, divergent replay from the same URL and
        # report metrics for whichever one the load balancer happened to pick.
        # Passing an application instance rather than an import string forecloses
        # multi-process mode anyway; this makes the reason visible.
        workers=1,
    )


@app.command()
def info(config: ConfigOption = None) -> None:
    """Print the resolved configuration and its fingerprint."""
    from football_insights.features.spec import DEFAULT_FEATURE_SPEC

    settings = _settings(config)
    _echo_json(
        {
            "config_fingerprint": settings.fingerprint(),
            "feature_schema": DEFAULT_FEATURE_SPEC.schema_hash,
            "n_features": DEFAULT_FEATURE_SPEC.n_features,
            "sequence_length": settings.window.sequence_length,
            "window": settings.window.model_dump(),
            "episode": settings.episode.model_dump(),
            "editorial": settings.editorial.model_dump(),
        }
    )


if __name__ == "__main__":  # pragma: no cover
    app()
