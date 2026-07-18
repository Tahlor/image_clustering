"""Serialization for clustering results and downstream handoff."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.models import ClusteringResult


def write_result(
    result: ClusteringResult,
    output_dir: Path,
    *,
    config: ClusterConfig | None = None,
) -> None:
    """Write a canonical manifest and flat diagnostics.

    Args:
        result: Clustering result to serialize.
        output_dir: Destination directory.
        config: Optional full configuration for `run.json`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clustering.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )

    rows = [comparison.to_dict() for comparison in result.comparisons]
    pair_scores_path = output_dir / "pair_scores.csv"
    if rows:
        with pair_scores_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        pair_scores_path.write_text("", encoding="utf-8")

    metadata = {
        "config_fingerprint": result.config_fingerprint,
        "config": asdict(config) if config is not None else None,
        "sequence_count": len({image.sequence_id for image in result.images}),
        "image_count": len(result.images),
        "candidate_pair_count": len(result.comparisons),
        "accepted_pair_count": sum(
            comparison.same_document for comparison in result.comparisons
        ),
        "cluster_count": len(result.clusters),
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_result(path: Path) -> ClusteringResult:
    """Load a canonical `clustering.json` manifest."""
    return ClusteringResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
