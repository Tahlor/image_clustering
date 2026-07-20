"""Public API for recovering unique pages and foreground inserts from clusters."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from image_clustering.clustering import (
    ClusterConfig,
    ClusteringResult,
    cluster_directory,
)

from .config import Config, load_default_config
from .io_utils import read_json, write_json
from .models import ImageRecord
from .pipeline import SequencePipeline

LOGGER = logging.getLogger(__name__)


def _cluster_signature(
    clustering: ClusteringResult,
    cluster_id: str,
    config: Config,
) -> str:
    """Hash crop configuration, cluster membership, and source metadata."""
    images = clustering.images_for(cluster_id)
    payload = {
        "crop_config": config.data,
        "clustering_config_fingerprint": clustering.config_fingerprint,
        "cluster_id": cluster_id,
        "images": [
            {
                "image_id": image.image_id,
                "path": str(image.path),
                "size": image.path.stat().st_size,
                "mtime_ns": image.path.stat().st_mtime_ns,
            }
            for image in images
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def crop_clustering_result(
    clustering: ClusteringResult,
    *,
    output_dir: Path,
    config: Config | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Recover unique pages and foreground inserts from clustering output.

    Clustering decides which captures show the same physical scene. Cropping
    then re-registers images within each accepted cluster at page level because
    crop localization requires stricter local alignment than cluster membership.

    Args:
        clustering: Upstream conservative document-view clustering result.
        output_dir: Directory for crops, review items, annotations, and manifests.
        config: Optional crop-localization configuration.
        show_progress: Whether to display progress for sufficiently large runs.

    Returns:
        JSON-serializable aggregate crop manifest.
    """
    resolved_config = config or load_default_config()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(resolved_config.get("io.resume", True))

    def process_cluster(cluster: Any) -> dict[str, Any] | None:
        images = clustering.images_for(cluster.cluster_id)
        if not images:
            return None
        parents = {image.path.parent.resolve() for image in images}
        if len(parents) != 1:
            raise ValueError(
                f"Cluster {cluster.cluster_id} crosses parent folders: "
                f"{sorted(parents)}"
            )
        folder = next(iter(parents))
        records = [
            ImageRecord(
                index=index,
                path=image.path.resolve(),
                folder=folder,
                name=image.path.name,
            )
            for index, image in enumerate(images)
        ]
        namespace = Path(cluster.sequence_id) / cluster.cluster_id
        manifest_path = output_dir / "cluster_manifests" / namespace / "manifest.json"
        signature = _cluster_signature(
            clustering=clustering,
            cluster_id=cluster.cluster_id,
            config=resolved_config,
        )
        if resume and manifest_path.is_file():
            cached = read_json(manifest_path)
            if isinstance(cached, dict) and cached.get("run_signature") == signature:
                LOGGER.info("Resuming crop cluster %s", cluster.cluster_id)
                return cached

        LOGGER.info(
            "Cropping %s (%d related captures)",
            cluster.cluster_id,
            len(records),
        )
        pipeline = SequencePipeline(config=resolved_config, output_dir=output_dir)
        result = pipeline.process(
            folder=folder,
            records=records,
            sequence_id=namespace.as_posix(),
            force_single_group=True,
        )
        result["upstream_cluster_id"] = cluster.cluster_id
        result["upstream_sequence_id"] = cluster.sequence_id
        result["run_signature"] = signature
        write_json(manifest_path, result)
        return result

    configured_workers = int(resolved_config.get("runtime.workers", 0))
    workers = (
        configured_workers
        if configured_workers > 0
        else min(4, max(1, os.cpu_count() or 1))
    )
    clusters = list(clustering.clusters)
    if workers == 1:
        processed = [
            process_cluster(cluster)
            for cluster in tqdm(
                clusters,
                desc="Crop clusters",
                unit="cluster",
                disable=not (show_progress and len(clusters) >= 8),
            )
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            processed = list(
                tqdm(
                    executor.map(process_cluster, clusters),
                    total=len(clusters),
                    desc="Crop clusters",
                    unit="cluster",
                    disable=not (show_progress and len(clusters) >= 8),
                )
            )
    cluster_results = [result for result in processed if result is not None]

    aggregate = {
        "schema_version": 1,
        "input_root": str(clustering.input_root) if clustering.input_root else None,
        "clustering_config_fingerprint": clustering.config_fingerprint,
        "crop_config_fingerprint": resolved_config.fingerprint(),
        "cluster_count": len(cluster_results),
        "submission_count": sum(
            len(result.get("submissions", [])) for result in cluster_results
        ),
        "clusters": cluster_results,
    }
    write_json(output_dir / "cropping.json", aggregate)
    return aggregate


def crop_directory(
    input_dir: Path,
    *,
    output_dir: Path,
    cluster_config: ClusterConfig | None = None,
    crop_config: Config | None = None,
    cache_dir: Path | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Cluster an image tree and recover unique crops in one call.

    Images are compared only within their immediate parent folder. Upstream
    clustering controls the nearby comparison window and cluster membership.

    Args:
        input_dir: Root containing one or more independent image folders.
        output_dir: Directory for clustering and crop outputs.
        cluster_config: Optional upstream clustering configuration.
        crop_config: Optional crop-localization configuration.
        cache_dir: Optional persistent clustering feature cache.
        show_progress: Whether to display progress bars for large runs.

    Returns:
        JSON-serializable aggregate crop manifest.
    """
    clustering = cluster_directory(
        input_dir=input_dir,
        config=cluster_config,
        cache_dir=cache_dir,
        show_progress=show_progress,
    )
    return crop_clustering_result(
        clustering=clustering,
        output_dir=output_dir,
        config=crop_config,
        show_progress=show_progress,
    )
