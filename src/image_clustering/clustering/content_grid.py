"""Pixel and coarse-grid analysis for document-specific content agreement."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_features import (
    _clean_mismatch,
    _ink_mask,
    local_dissimilarity,
)
from image_clustering.clustering.content_geometry import _tile_bounds
from image_clustering.clustering.content_models import ContentGrid


def compute_content_grid(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    config: ClusterConfig,
) -> ContentGrid | None:
    """Compute tolerant ink mismatch and robust residual tile maps."""
    core = cv2.erode(valid_mask, np.ones((9, 9), np.uint8)) > 0
    if int(core.sum()) < 1000:
        return None

    reference_ink = _ink_mask(reference, core, config)
    aligned_ink = _ink_mask(aligned, core, config)
    tolerance = max(1, round(min(reference.shape) * config.ink_tolerance_fraction))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerance + 1, 2 * tolerance + 1),
    )
    reference_dilated = cv2.dilate(reference_ink.astype(np.uint8), kernel) > 0
    aligned_dilated = cv2.dilate(aligned_ink.astype(np.uint8), kernel) > 0
    mismatch = (
        (reference_ink & ~aligned_dilated) | (aligned_ink & ~reference_dilated)
    ) & core
    mismatch, component_areas = _clean_mismatch(mismatch, config)
    ink_union = (reference_ink | aligned_ink) & core
    unmatched_fraction = float(mismatch.sum() / max(core.sum(), 1))
    unmatched_union_fraction = float(mismatch.sum() / max(ink_union.sum(), 1))
    largest_component_fraction = max(component_areas, default=0) / max(core.sum(), 1)

    dissimilarity = local_dissimilarity(reference, aligned)
    rows = config.content_tile_rows
    columns = config.content_tile_columns
    scores = np.full((rows, columns), np.nan, dtype=np.float32)
    valid_tiles = np.zeros((rows, columns), dtype=bool)
    ink_tile_mismatch = np.zeros((rows, columns), dtype=bool)
    for row in range(rows):
        for column in range(columns):
            x0, y0, x1, y1 = _tile_bounds(
                row,
                column,
                reference.shape,
                rows,
                columns,
            )
            tile_core = core[y0:y1, x0:x1]
            if tile_core.mean() < 0.5:
                continue
            values = dissimilarity[y0:y1, x0:x1][tile_core]
            if values.size < 20:
                continue
            tail_count = max(
                1, round(config.residual_upper_tail_fraction * values.size)
            )
            upper_tail = float(np.partition(values, -tail_count)[-tail_count:].mean())
            scores[row, column] = 0.4 * float(values.mean()) + 0.6 * upper_tail
            valid_tiles[row, column] = True
            tile_mismatch = mismatch[y0:y1, x0:x1][tile_core]
            tile_union = ink_union[y0:y1, x0:x1][tile_core]
            mismatch_union = float(tile_mismatch.sum() / max(tile_union.sum(), 10))
            mismatch_valid = float(tile_mismatch.mean())
            ink_tile_mismatch[row, column] = (
                mismatch_union >= config.ink_tile_union_threshold
                and mismatch_valid >= config.ink_tile_valid_threshold
            )

    valid_scores = scores[valid_tiles]
    if valid_scores.size:
        ordered = np.sort(valid_scores)
        stable_count = max(4, round(config.residual_stable_fraction * len(ordered)))
        stable = ordered[:stable_count]
        baseline = float(np.median(stable))
        mad = float(np.median(np.abs(stable - baseline)))
        scale = max(1.4826 * mad, config.residual_min_scale)
        zscores = np.zeros_like(scores)
        zscores[valid_tiles] = (scores[valid_tiles] - baseline) / scale
        changed = valid_tiles & (
            (zscores >= config.residual_changed_z_threshold)
            & (scores >= config.residual_changed_min_absolute)
        )
    else:
        zscores = np.zeros_like(scores)
        changed = np.zeros_like(valid_tiles)
    changed = cv2.morphologyEx(
        changed.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    changed &= valid_tiles
    residual_changed = float(changed.sum() / max(valid_tiles.sum(), 1))
    ink_tiles_changed = float(
        (ink_tile_mismatch & valid_tiles).sum() / max(valid_tiles.sum(), 1)
    )
    return ContentGrid(
        core=core,
        mismatch=mismatch,
        ink_union=ink_union,
        component_areas=component_areas,
        unmatched_fraction=unmatched_fraction,
        unmatched_union_fraction=unmatched_union_fraction,
        largest_component_fraction=float(largest_component_fraction),
        scores=scores,
        valid_tiles=valid_tiles,
        ink_tile_mismatch=ink_tile_mismatch,
        zscores=zscores,
        changed=changed,
        residual_tiles_changed_fraction=residual_changed,
        ink_mismatch_tiles_fraction=ink_tiles_changed,
    )
