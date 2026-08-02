"""Dataset acquisition.

The Metrica sample data is downloaded at setup time and never committed. Two
reasons, both worth stating plainly:

* it carries **no formal licence**. The publisher asks only that users "be
  responsible" and "acknowledge the source", which is permission to use but not
  a grant to redistribute. Vendoring it into a public repository would assume a
  right that was never given.
* it is roughly 180 MB, which does not belong in git regardless.

Every file is recorded in a manifest with its SHA-256, so a processed dataset
can be traced to exactly the bytes it came from, and a partial or corrupted
download fails loudly instead of silently training on half a match.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from football_insights.errors import DataValidationError
from football_insights.types import JsonDict

BASE_URL: Final = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"

#: Attribution required by the publisher, reproduced wherever the data is used.
ATTRIBUTION: Final = (
    "Tracking and event data provided by Metrica Sports "
    "(https://github.com/metrica-sports/sample-data). "
    "Used with acknowledgement; not redistributed by this project."
)


@dataclass(frozen=True, slots=True)
class MatchFiles:
    """The files making up one sample match."""

    match_id: str
    source_format: str
    files: dict[str, str]

    def paths(self, root: Path) -> dict[str, Path]:
        """Local destination for each file."""
        return {key: root / self.match_id / Path(name).name for key, name in self.files.items()}


#: The three matches published by Metrica. Games 1 and 2 are CSV; game 3 uses
#: the EPTS/FIFA format with a metadata XML and a JSON event file.
AVAILABLE_MATCHES: Final[tuple[MatchFiles, ...]] = (
    MatchFiles(
        match_id="Sample_Game_1",
        source_format="metrica_csv",
        files={
            "home": "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
            "away": "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
            "events": "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        },
    ),
    MatchFiles(
        match_id="Sample_Game_2",
        source_format="metrica_csv",
        files={
            "home": "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
            "away": "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
            "events": "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        },
    ),
    MatchFiles(
        match_id="Sample_Game_3",
        source_format="metrica_epts",
        files={
            "tracking": "Sample_Game_3/Sample_Game_3_tracking.txt",
            "metadata": "Sample_Game_3/Sample_Game_3_metadata.xml",
            "events": "Sample_Game_3/Sample_Game_3_events.json",
        },
    ),
)

MATCHES_BY_ID: Final = {m.match_id: m for m in AVAILABLE_MATCHES}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, timeout: float = 120.0) -> None:
    """Download one file, writing atomically.

    A partial download is written to a temporary path and moved into place only
    on success, so an interrupted run never leaves a truncated file that looks
    complete on the next attempt.

    Args:
        url: Source URL.
        destination: Local path.
        timeout: Socket timeout in seconds.

    Raises:
        DataValidationError: If the download fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        # The URL is not user-supplied: callers pass a constant from
        # DATASET_FILES, which is fixed at https://raw.githubusercontent.com.
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                msg = f"{url} returned HTTP {response.status}"
                raise DataValidationError(msg)
            with temporary.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        msg = f"failed to download {url}: {exc}"
        raise DataValidationError(msg) from exc
    temporary.replace(destination)


def acquire_match(
    match: MatchFiles,
    root: Path,
    force: bool = False,
) -> JsonDict:
    """Fetch one match and record its fingerprints.

    Args:
        match: The match to fetch.
        root: Raw data directory.
        force: Re-download even if the files are already present.

    Returns:
        Manifest entry for the match.
    """
    entry: JsonDict = {
        "match_id": match.match_id,
        "source_format": match.source_format,
        "source": f"{BASE_URL}/{match.match_id}",
        "attribution": ATTRIBUTION,
        "files": {},
    }
    files: JsonDict = entry["files"]
    for key, relative in match.files.items():
        destination = root / match.match_id / Path(relative).name
        if force or not destination.is_file():
            download(f"{BASE_URL}/{relative}", destination)
        files[key] = {
            "path": str(destination.relative_to(root)),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    return entry


def acquire(
    root: Path,
    match_ids: list[str] | None = None,
    force: bool = False,
) -> JsonDict:
    """Fetch the sample dataset and write a manifest.

    Args:
        root: Raw data directory.
        match_ids: Matches to fetch; all three when omitted.
        force: Re-download even if files are present.

    Returns:
        The manifest that was written.

    Raises:
        KeyError: If an unknown match id is requested.
    """
    selected = [MATCHES_BY_ID[m] for m in match_ids] if match_ids else list(AVAILABLE_MATCHES)
    root.mkdir(parents=True, exist_ok=True)
    manifest: JsonDict = {
        "dataset": "metrica-sports/sample-data",
        "attribution": ATTRIBUTION,
        "licence": (
            "No formal licence is published. The source asks users to be responsible "
            "and to acknowledge the source. Not redistributed by this project; "
            "downloaded locally at setup time."
        ),
        "matches": [acquire_match(m, root, force) for m in selected],
    }
    manifest["fingerprint"] = dataset_fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def dataset_fingerprint(manifest: JsonDict) -> str:
    """Single hash identifying the exact dataset contents.

    Recorded in model metadata so results can be tied to the data behind them.

    Args:
        manifest: A manifest produced by :func:`acquire`.

    Returns:
        Short hex digest over every file hash, in a stable order.
    """
    digests: list[str] = []
    for match in manifest.get("matches", []):
        for key in sorted(match["files"]):
            digests.append(f"{match['match_id']}:{key}:{match['files'][key]['sha256']}")
    return hashlib.sha256("|".join(sorted(digests)).encode()).hexdigest()[:16]


def load_manifest(root: Path) -> JsonDict:
    """Read the manifest, verifying the files still match their recorded hashes.

    Args:
        root: Raw data directory.

    Returns:
        The manifest.

    Raises:
        DataValidationError: If the manifest is missing or a file has changed.
    """
    path = root / "manifest.json"
    if not path.is_file():
        msg = (
            f"no dataset manifest at {path}. Run `football-insights acquire` "
            "to download the sample data."
        )
        raise DataValidationError(msg)
    manifest: JsonDict = json.loads(path.read_text())
    for match in manifest["matches"]:
        for key, info in match["files"].items():
            file_path = root / info["path"]
            if not file_path.is_file():
                msg = f"{match['match_id']}: {key} file missing at {file_path}"
                raise DataValidationError(msg)
            actual = sha256_file(file_path)
            if actual != info["sha256"]:
                msg = (
                    f"{match['match_id']}: {key} file has changed since it was "
                    f"downloaded (expected {info['sha256'][:12]}, found {actual[:12]}). "
                    "Re-run acquisition with --force."
                )
                raise DataValidationError(msg)
    return manifest
