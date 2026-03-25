from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactProvenance:
    """Stable metadata for one artifact file in a demo bundle."""

    name: str
    size_bytes: int
    sha256: str


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> ArtifactProvenance:
    """Capture reproducibility metadata for a single artifact file."""
    stat = path.stat()
    return ArtifactProvenance(
        name=path.name,
        size_bytes=stat.st_size,
        sha256=hash_file(path),
    )


def canonical_json_digest(payload: Any) -> str:
    """Hash a JSON-serializable payload in a stable, canonical form."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def bundle_inventory(
    artifact_dir: Path, filenames: tuple[str, ...]
) -> tuple[ArtifactProvenance, ...]:
    """Collect provenance metadata for an ordered bundle of files."""
    return tuple(file_provenance(artifact_dir / filename) for filename in filenames)


def bundle_fingerprint(inventory: tuple[ArtifactProvenance, ...]) -> str:
    """Return a deterministic fingerprint for a bundle inventory."""
    return canonical_json_digest(
        {
            "algorithm": "sha256",
            "artifacts": [
                {
                    "name": item.name,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in inventory
            ],
        }
    )


def runtime_environment_snapshot() -> dict[str, str]:
    """Capture the local runtime context used to interpret a demo bundle."""

    def _version(package_name: str) -> str:
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "batteryops": _version("batteryops"),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scipy": _version("scipy"),
        "scikit-learn": _version("scikit-learn"),
        "joblib": _version("joblib"),
    }
