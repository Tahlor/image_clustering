"""Aggregate physical-occlusion candidates into public content metrics."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_geometry import _tile_bounds
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
    rows, columns = grid.valid_tiles.shape
    for candidate in selected:
        bbox = candidate["bbox"]
        assert isinstance(bbox, tuple)
        x0, y0, x1, y1 = bbox
        support = candidate["support"]
        assert isinstance(support, np.ndarray)
        candidate_tile_mask |= support
        # Convert only the connected residual support to pixels. A narrow
        # dilation absorbs imperfect/skewed seams without filling the entire
        # component bounding box.
        for row, column in zip(*np.where(support), strict=True):
            tx0, ty0, tx1, ty1 = _tile_bounds(
                row,
                column,
                reference.shape,
                rows,
                columns,
            )
            candidate_mask[ty0:ty1, tx0:tx1] = True
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

    if candidate_mask.any():
        seam_tolerance = max(
            3,
            (round(min(reference.shape) * 0.008) // 2) * 2 + 1,
        )
        candidate_mask = cv2.dilate(
            candidate_mask.astype(np.uint8),
            np.ones((seam_tolerance, seam_tolerance), np.uint8),
        ).astype(bool)
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

    inside = grid.core & candidate_mask
    inside_mismatch = int(grid.mismatch[inside].sum())
    total_mismatch = int(grid.mismatch[grid.core].sum())
    inside_unmatched_union_fraction = float(
        inside_mismatch / max(grid.ink_union[inside].sum(), 10)
    )
    occlusion_ink_mismatch_capture = float(
        inside_mismatch / max(total_mismatch, 1)
    )

    outside = grid.core & ~candidate_mask
    outside_unmatched_fraction = (
        float(grid.mismatch[outside].mean()) if outside.any() else 1.0
    )
    outside_unmatched_union_fraction = float(
        grid.mismatch[outside].sum() / max(grid.ink_union[outside].sum(), 10)
    )
    localization_contrast = max(
        0.0,
        inside_unmatched_union_fraction - outside_unmatched_union_fraction,
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
        inside_unmatched_ink_union_fraction=(
            inside_unmatched_union_fraction
        ),
        occlusion_ink_mismatch_capture=occlusion_ink_mismatch_capture,
        occlusion_localization_contrast=localization_contrast,
    )
