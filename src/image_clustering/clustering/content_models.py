"""Models for document-specific content agreement analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContentMetrics:
    """Content agreement and occlusion-explanation diagnostics."""

    unmatched_ink_fraction: float
    unmatched_ink_union_fraction: float
    ink_mismatch_tiles_fraction: float
    coherent_ink_component_count: int
    largest_ink_component_fraction: float
    residual_tiles_changed_fraction: float
    occlusion_candidate_count: int
    occlusion_area_fraction: float
    occlusion_residual_capture: float
    occlusion_rectangularity: float
    occlusion_boundary_score: float
    occlusion_material_fraction: float
    occlusion_material_median: float
    outside_unmatched_ink_fraction: float
    outside_unmatched_ink_union_fraction: float
    outside_ink_mismatch_tiles_fraction: float
    full_page_occlusion_count: int
    shallow_occlusion_count: int
    page_count: int


@dataclass(frozen=True)
class ContentGrid:
    """Intermediate pixel and tile-level content analysis."""

    core: np.ndarray
    mismatch: np.ndarray
    ink_union: np.ndarray
    component_areas: list[int]
    unmatched_fraction: float
    unmatched_union_fraction: float
    largest_component_fraction: float
    scores: np.ndarray
    valid_tiles: np.ndarray
    ink_tile_mismatch: np.ndarray
    zscores: np.ndarray
    changed: np.ndarray
    residual_tiles_changed_fraction: float
    ink_mismatch_tiles_fraction: float
