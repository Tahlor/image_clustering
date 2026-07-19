"""Document-specific ink agreement and physical-occlusion analysis."""

from __future__ import annotations

import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_features import local_dissimilarity
from image_clustering.clustering.content_grid import compute_content_grid
from image_clustering.clustering.content_models import ContentMetrics
from image_clustering.clustering.content_pages import select_page_candidates
from image_clustering.clustering.content_summary import build_content_metrics

__all__ = ["ContentMetrics", "analyze_content", "local_dissimilarity"]


def _empty_metrics() -> ContentMetrics:
    return ContentMetrics(
        1.0,
        1.0,
        1.0,
        0,
        1.0,
        1.0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0,
        1,
    )


def analyze_content(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    config: ClusterConfig,
) -> ContentMetrics:
    """Measure exact ink agreement and whether a physical occluder explains change."""
    grid = compute_content_grid(reference, aligned, valid_mask, config)
    if grid is None:
        return _empty_metrics()
    selected, page_regions, gutter = select_page_candidates(
        reference=reference,
        aligned=aligned,
        grid=grid,
        config=config,
    )
    return build_content_metrics(
        reference=reference,
        aligned=aligned,
        grid=grid,
        selected=selected,
        page_regions=page_regions,
        gutter=gutter,
        config=config,
    )
