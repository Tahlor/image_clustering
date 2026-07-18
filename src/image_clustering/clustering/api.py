"""Public orchestration API for document-view clustering."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import cv2
from tqdm import tqdm

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.discovery import discover_sequences, make_image_items
from image_clustering.clustering.features import extract_features
from image_clustering.clustering.graph import build_clusters
from image_clustering.clustering.models import (
    ClusteringResult,
    ImageCluster,
    ImageFeatures,
    ImageItem,
    PairComparison,
)
from image_clustering.clustering.scoring import score_pair

T = TypeVar("T")
R = TypeVar("R")


def _worker_count(config: ClusterConfig) -> int:
    if config.workers > 0:
        return config.workers
    return min(8, max(1, os.cpu_count() or 1))


def _ordered_map(
    function: Callable[[T], R],
    items: Sequence[T],
    workers: int,
    description: str,
    unit: str,
    show_progress: bool,
) -> list[R]:
    show_bar = show_progress and len(items) >= 8
    if workers == 1:
        return [
            function(item)
            for item in tqdm(
                items,
                desc=description,
                leave=False,
                unit=unit,
                disable=not show_bar,
            )
        ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(function, items),
                total=len(items),
                desc=description,
                leave=False,
                unit=unit,
                disable=not show_bar,
            )
        )


def _score_sequence(
    features: list[ImageFeatures],
    config: ClusterConfig,
    workers: int,
    show_progress: bool,
) -> list[PairComparison]:
    jobs = []
    for first_index in range(len(features)):
        stop = min(len(features), first_index + config.max_gap + 1)
        jobs.extend(
            (first_index, second_index)
            for second_index in range(first_index + 1, stop)
        )

    def run(job: tuple[int, int]) -> PairComparison:
        first_index, second_index = job
        return score_pair(
            previous=features[first_index],
            current=features[second_index],
            index_gap=second_index - first_index,
            config=config,
        )

    sequence_id = features[0].image.sequence_id
    return _ordered_map(
        function=run,
        items=jobs,
        workers=workers,
        description=f"Pairs {sequence_id}",
        unit="pair",
        show_progress=show_progress,
    )


def _cluster_sequence(
    images: tuple[ImageItem, ...],
    config: ClusterConfig,
    cache_dir: Path | None,
    workers: int,
    cluster_id_start: int,
    show_progress: bool,
) -> tuple[list[ImageCluster], list[PairComparison]]:
    features = _ordered_map(
        function=lambda image: extract_features(
            image=image,
            config=config,
            cache_dir=cache_dir,
        ),
        items=images,
        workers=workers,
        description=f"Features {images[0].sequence_id}",
        unit="image",
        show_progress=show_progress,
    )
    comparisons = _score_sequence(
        features=features,
        config=config,
        workers=workers,
        show_progress=show_progress,
    )
    clusters = build_clusters(
        sequence_id=images[0].sequence_id,
        image_ids=[image.image_id for image in images],
        comparisons=comparisons,
        cluster_id_start=cluster_id_start,
    )
    return clusters, comparisons


def cluster_images(
    image_paths: Sequence[Path],
    *,
    sequence_id: str = ".",
    config: ClusterConfig | None = None,
    cache_dir: Path | None = None,
    show_progress: bool = False,
) -> ClusteringResult:
    """Cluster one explicitly ordered image sequence.

    Args:
        image_paths: Images in sequence order. This function does not sort them.
        sequence_id: Stable identifier for the independent sequence.
        config: Optional clustering configuration.
        cache_dir: Optional persistent feature cache.
        show_progress: Whether to display progress for sufficiently large jobs.

    Returns:
        A complete result containing clusters, images, and pair diagnostics.
    """
    resolved_config = config or ClusterConfig()
    images = make_image_items(image_paths=image_paths, sequence_id=sequence_id)
    workers = _worker_count(config=resolved_config)
    cv2.setNumThreads(1)
    clusters, comparisons = _cluster_sequence(
        images=images,
        config=resolved_config,
        cache_dir=cache_dir,
        workers=workers,
        cluster_id_start=1,
        show_progress=show_progress,
    )
    return ClusteringResult(
        config_fingerprint=resolved_config.fingerprint(),
        images=images,
        clusters=tuple(clusters),
        comparisons=tuple(comparisons),
    )


def cluster_directory(
    input_dir: Path,
    *,
    config: ClusterConfig | None = None,
    cache_dir: Path | None = None,
    show_progress: bool = False,
) -> ClusteringResult:
    """Cluster every independent filename-ordered folder below a root.

    Args:
        input_dir: Root containing images in one or more folders.
        config: Optional clustering configuration.
        cache_dir: Optional persistent feature cache.
        show_progress: Whether to display progress for sufficiently large jobs.

    Returns:
        Aggregate clustering result. Images in different folders are never
        compared or placed in the same cluster.
    """
    input_dir = input_dir.resolve()
    resolved_config = config or ClusterConfig()
    sequences = discover_sequences(input_dir=input_dir)
    workers = _worker_count(config=resolved_config)
    cv2.setNumThreads(1)

    all_images: list[ImageItem] = []
    all_clusters: list[ImageCluster] = []
    all_comparisons: list[PairComparison] = []
    next_cluster_id = 1
    sequence_iterator = tqdm(
        sequences,
        desc="Sequences",
        unit="folder",
        disable=not (show_progress and len(sequences) >= 2),
    )
    for images in sequence_iterator:
        clusters, comparisons = _cluster_sequence(
            images=images,
            config=resolved_config,
            cache_dir=cache_dir,
            workers=workers,
            cluster_id_start=next_cluster_id,
            show_progress=show_progress,
        )
        all_images.extend(images)
        all_clusters.extend(clusters)
        all_comparisons.extend(comparisons)
        next_cluster_id += len(clusters)

    return ClusteringResult(
        config_fingerprint=resolved_config.fingerprint(),
        input_root=input_dir,
        images=tuple(all_images),
        clusters=tuple(all_clusters),
        comparisons=tuple(all_comparisons),
    )
