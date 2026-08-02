"""Exact, restartable cache for completed pair comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.models import ImageFeatures, PairComparison

_PAIR_CACHE_VERSION = 1
_NON_SCORING_CONFIG_FIELDS = {
    "cache_features",
    "cache_pairs",
    "cache_working_images",
    "feature_cache_compressed",
    "workers",
    "max_gap",
    "max_cluster_size",
}


@lru_cache(maxsize=1)
def _algorithm_fingerprint() -> str:
    """Hash every clustering source file that can affect pair evidence or decisions."""
    module_dir = Path(__file__).resolve().parent
    names: set[Path] = {
        module_dir / "_config_schema.py",
        module_dir / "candidate_scoring.py",
        module_dir / "config.py",
        module_dir / "features.py",
        module_dir / "models.py",
        module_dir / "pair_cache.py",
        module_dir / "registration.py",
        module_dir / "scoring.py",
        module_dir / "scoring_decision.py",
    }
    names.update(module_dir.glob("content*.py"))
    digest = hashlib.sha256()
    for path in sorted(names):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _scoring_config_fingerprint(config: ClusterConfig) -> str:
    values = asdict(config)
    for field_name in _NON_SCORING_CONFIG_FIELDS:
        values.pop(field_name, None)
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_identity(features: ImageFeatures) -> dict[str, object]:
    path = features.image.path.resolve()
    stat = path.stat()
    return {
        "image_id": features.image.image_id,
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def pair_cache_key(
    previous: ImageFeatures,
    current: ImageFeatures,
    index_gap: int,
    config: ClusterConfig,
) -> str:
    """Return a stable key for an exact pair-scoring result."""
    payload = {
        "version": _PAIR_CACHE_VERSION,
        "algorithm": _algorithm_fingerprint(),
        "config": _scoring_config_fingerprint(config),
        "previous": _source_identity(previous),
        "current": _source_identity(current),
        "index_gap": index_gap,
        "sequence_id": previous.image.sequence_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "pair_comparisons" / key[:2] / f"{key}.json"


def load_pair_comparison(
    cache_dir: Path,
    key: str,
) -> PairComparison | None:
    """Load one complete pair result, ignoring incomplete or corrupt entries."""
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != _PAIR_CACHE_VERSION:
            return None
        return PairComparison.from_dict(value["comparison"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def save_pair_comparison(
    cache_dir: Path,
    key: str,
    comparison: PairComparison,
) -> None:
    """Atomically checkpoint one complete pair result."""
    path = _cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    payload = {
        "schema_version": _PAIR_CACHE_VERSION,
        "comparison": comparison.to_dict(),
    }
    try:
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "load_pair_comparison",
    "pair_cache_key",
    "save_pair_comparison",
]
