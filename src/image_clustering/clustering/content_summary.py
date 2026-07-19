"""Aggregate physical-occlusion candidates into public content metrics."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_models import ContentGrid, ContentMetrics
from image_clustering.clustering.content_pages import Candidate, PageRegion


def build_content_metrics(
    reference: np.ndarray,
    aligned: np.ndarray,
    grid: ContentGrid,
    selected: list[Candidate],
    page_regions: list[PageRegion],
    gutter: int | None,
    config: ClusterConfig,
) -> ContentMetrics:
    """Measure coverage, material change, and outside-occlusion ink agreement."""
    candidate_mask = np.zeros_like(grid.core)
    candidate_tile_mask = np.zeros_like(grid.valid_tiles)
    rectangularities: list[float] = []
    boundaries: list[float] = []
    full_page_count = 0
    shallow_count = 0
    for candidate in selected:
        bbox = candidate["bbox"]
        assert isinstance(bbox, tuple)
        x0, y0, x1, y1 = bbox
        candidate_mask[y0:y1, x0:x1] = True
        support = candidate["support"]
        assert isinstance(support, np.ndarray)
        candidate_tile_mask |= support
        rectangularities.append(float(candidate["rectangularity"]))
        boundaries.append(float(candidate["boundary"]))
        full_page_count += int(candidate["full_page"])
        page_width = (
            reference.shape[1]
            if gutter is None
            else max(gutter, reference.shape[1] - gutter)
        )
        page_height = reference.shape[0]
        if (y1 - y0) / max(
            page_height, 1
        ) <= config.occlusion_shallow_max_height_fraction and (x1 - x0) / max(
            page_width, 1
        ) >= config.occlusion_shallow_min_width_fraction:
            shallow_count += 1

    candidate_mask &= grid.core
    candidate_area_fraction = float(candidate_mask.sum() / max(grid.core.sum(), 1))
    excess = np.maximum(grid.zscores, 0.0)
    residual_capture = float(
        excess[candidate_tile_mask].sum() / max(excess[grid.valid_tiles].sum(), 1e-6)
    )
    material_sigma = max(3.0, min(reference.shape) / 120.0)
    reference_material = cv2.GaussianBlur(
        reference.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=material_sigma,
    )
    aligned_material = cv2.GaussianBlur(
        aligned.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=material_sigma,
    )
    material_difference = np.abs(reference_material - aligned_material)
    material_values = material_difference[candidate_mask]
    material_fraction = (
        float(np.mean(material_values >= 0.04)) if material_values.size else 0.0
    )
    material_median = float(np.median(material_values)) if material_values.size else 0.0

    outside = grid.core & ~candidate_mask
    outside_unmatched_fraction = (
        float(grid.mismatch[outside].mean()) if outside.any() else 1.0
    )
    outside_unmatched_union_fraction = float(
        grid.mismatch[outside].sum() / max(grid.ink_union[outside].sum(), 10)
    )
    outside_tile_mask = grid.valid_tiles & ~candidate_tile_mask
    outside_ink_tiles_fraction = float(
        (grid.ink_tile_mismatch & outside_tile_mask).sum()
        / max(outside_tile_mask.sum(), 1)
    )
    return ContentMetrics(
        unmatched_ink_fraction=grid.unmatched_fraction,
        unmatched_ink_union_fraction=grid.unmatched_union_fraction,
        ink_mismatch_tiles_fraction=grid.ink_mismatch_tiles_fraction,
        coherent_ink_component_count=len(grid.component_areas),
        largest_ink_component_fraction=grid.largest_component_fraction,
        residual_tiles_changed_fraction=grid.residual_tiles_changed_fraction,
        occlusion_candidate_count=len(selected),
        occlusion_area_fraction=candidate_area_fraction,
        occlusion_residual_capture=residual_capture,
        occlusion_rectangularity=(
            float(np.mean(rectangularities)) if rectangularities else 0.0
        ),
        occlusion_boundary_score=max(boundaries, default=0.0),
        occlusion_material_fraction=material_fraction,
        occlusion_material_median=material_median,
        outside_unmatched_ink_fraction=outside_unmatched_fraction,
        outside_unmatched_ink_union_fraction=outside_unmatched_union_fraction,
        outside_ink_mismatch_tiles_fraction=outside_ink_tiles_fraction,
        full_page_occlusion_count=full_page_count,
        shallow_occlusion_count=shallow_count,
        page_count=len(page_regions),
    )
