"""Configuration for conservative physical-document clustering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClusterConfig:
    """Thresholds and resource limits for document-view clustering.

    The defaults intentionally favor false negatives. A rejected pair can be
    handled as two clusters downstream; a false merge can conflate different
    people or records.
    """

    max_gap: int = 3
    max_working_dimension: int = 900
    max_features: int = 2500
    sift_contrast_threshold: float = 0.025
    ratio_test: float = 0.72
    ransac_reprojection_px: float = 3.5
    min_inliers: int = 12
    min_inlier_ratio: float = 0.18
    min_grid_cells: int = 4
    min_x_span: float = 0.30
    min_y_span: float = 0.25
    change_threshold: float = 0.15
    standard_min_feature_overlap: float = 0.11
    standard_max_changed_fraction: float = 0.62
    heavy_min_feature_overlap: float = 0.18
    heavy_max_changed_fraction: float = 0.82
    exceptional_min_feature_overlap: float = 0.28
    exceptional_max_changed_fraction: float = 0.90
    min_valid_fraction: float = 0.65
    tile_rows: int = 6
    tile_columns: int = 8
    tile_changed_threshold: float = 0.20

    # Document-specific ink agreement. SIFT establishes registration; these
    # thresholds decide whether the filled content is actually the same.
    ink_background_sigma_fraction: float = 0.018
    ink_gradient_weight: float = 0.16
    ink_min_response: float = 0.085
    ink_min_component_fraction: float = 0.000003
    ink_mismatch_min_component_fraction: float = 0.000008
    ink_tolerance_fraction: float = 0.0025
    ink_tile_union_threshold: float = 0.08
    ink_tile_valid_threshold: float = 0.004
    content_tile_rows: int = 10
    content_tile_columns: int = 14

    # Near duplicates require essentially exact document-specific ink.
    duplicate_min_feature_overlap: float = 0.08
    duplicate_max_changed_fraction: float = 0.35
    duplicate_max_unmatched_ink_fraction: float = 0.0025
    duplicate_max_unmatched_ink_union_fraction: float = 0.02
    duplicate_max_ink_mismatch_tiles_fraction: float = 0.10

    # Non-duplicate pairs are accepted only when a physical occluder explains
    # nearly all meaningful disagreement.
    residual_upper_tail_fraction: float = 0.20
    residual_stable_fraction: float = 0.58
    residual_min_scale: float = 0.012
    residual_changed_z_threshold: float = 2.4
    residual_changed_min_absolute: float = 0.075
    occlusion_min_feature_overlap: float = 0.10
    occlusion_min_component_tiles: int = 2
    occlusion_min_page_area_fraction: float = 0.025
    occlusion_padding_x_fraction: float = 0.05
    occlusion_padding_y_fraction: float = 0.08
    occlusion_full_page_tile_fraction: float = 0.90
    occlusion_full_page_ink_tile_fraction: float = 0.60
    occlusion_full_page_material_fraction: float = 0.30
    occlusion_full_page_low_changed_fraction: float = 0.15
    occlusion_full_page_min_ink_mismatch_fraction: float = 0.45
    occlusion_min_area_fraction: float = 0.025
    occlusion_max_area_fraction: float = 0.85
    occlusion_min_residual_capture: float = 0.45
    occlusion_strong_boundary_min_capture: float = 0.30
    occlusion_min_boundary_score: float = 0.90
    occlusion_strong_boundary_score: float = 1.45
    occlusion_min_material_fraction: float = 0.12
    occlusion_large_clean_max_area_fraction: float = 0.95
    occlusion_large_clean_max_unmatched_ink_union_fraction: float = 0.08
    occlusion_max_outside_unmatched_ink_fraction: float = 0.028
    occlusion_max_outside_unmatched_ink_union_fraction: float = 0.115
    occlusion_max_outside_ink_mismatch_tiles_fraction: float = 0.30
    occlusion_shallow_max_height_fraction: float = 0.30
    occlusion_shallow_min_width_fraction: float = 0.50
    gutter_min_aspect_ratio: float = 1.15
    gutter_search_min_fraction: float = 0.34
    gutter_search_max_fraction: float = 0.66
    gutter_min_prominence: float = 0.18

    # A registered rejection blocks a transitive graph merge only when both
    # ink and residual disagreement are distributed rather than occlusion-like.
    contradiction_min_ink_mismatch_tiles_fraction: float = 0.25
    contradiction_min_residual_tiles_changed_fraction: float = 0.18
    contradiction_min_unmatched_ink_union_fraction: float = 0.06
    contradiction_overwhelming_ink_tiles_fraction: float = 0.70
    contradiction_overwhelming_unmatched_ink_union_fraction: float = 0.25
    contradiction_overwhelming_outside_ink_union_fraction: float = 0.20

    cache_features: bool = True
    workers: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_gap < 1:
            raise ValueError("max_gap must be at least 1")
        if self.max_working_dimension < 128:
            raise ValueError("max_working_dimension must be at least 128")
        if self.max_features < self.min_inliers:
            raise ValueError("max_features must be at least min_inliers")
        if self.workers < 0:
            raise ValueError("workers cannot be negative")
        if self.content_tile_rows < 2 or self.content_tile_columns < 2:
            raise ValueError("content tile grid must be at least 2 by 2")
        if self.occlusion_min_component_tiles < 1:
            raise ValueError("occlusion_min_component_tiles must be positive")
        unit_interval_fields = (
            "ratio_test",
            "min_inlier_ratio",
            "min_x_span",
            "min_y_span",
            "change_threshold",
            "standard_min_feature_overlap",
            "standard_max_changed_fraction",
            "heavy_min_feature_overlap",
            "heavy_max_changed_fraction",
            "exceptional_min_feature_overlap",
            "exceptional_max_changed_fraction",
            "min_valid_fraction",
            "tile_changed_threshold",
            "ink_background_sigma_fraction",
            "ink_gradient_weight",
            "ink_min_response",
            "ink_min_component_fraction",
            "ink_mismatch_min_component_fraction",
            "ink_tolerance_fraction",
            "ink_tile_union_threshold",
            "ink_tile_valid_threshold",
            "duplicate_min_feature_overlap",
            "duplicate_max_changed_fraction",
            "duplicate_max_unmatched_ink_fraction",
            "duplicate_max_unmatched_ink_union_fraction",
            "duplicate_max_ink_mismatch_tiles_fraction",
            "residual_upper_tail_fraction",
            "residual_stable_fraction",
            "residual_min_scale",
            "residual_changed_min_absolute",
            "occlusion_min_feature_overlap",
            "occlusion_min_page_area_fraction",
            "occlusion_padding_x_fraction",
            "occlusion_padding_y_fraction",
            "occlusion_full_page_tile_fraction",
            "occlusion_full_page_ink_tile_fraction",
            "occlusion_full_page_material_fraction",
            "occlusion_full_page_low_changed_fraction",
            "occlusion_full_page_min_ink_mismatch_fraction",
            "occlusion_min_area_fraction",
            "occlusion_max_area_fraction",
            "occlusion_min_residual_capture",
            "occlusion_strong_boundary_min_capture",
            "occlusion_min_material_fraction",
            "occlusion_large_clean_max_area_fraction",
            "occlusion_large_clean_max_unmatched_ink_union_fraction",
            "occlusion_max_outside_unmatched_ink_fraction",
            "occlusion_max_outside_unmatched_ink_union_fraction",
            "occlusion_max_outside_ink_mismatch_tiles_fraction",
            "occlusion_shallow_max_height_fraction",
            "occlusion_shallow_min_width_fraction",
            "gutter_search_min_fraction",
            "gutter_search_max_fraction",
            "gutter_min_prominence",
            "contradiction_min_ink_mismatch_tiles_fraction",
            "contradiction_min_residual_tiles_changed_fraction",
            "contradiction_min_unmatched_ink_union_fraction",
            "contradiction_overwhelming_ink_tiles_fraction",
            "contradiction_overwhelming_unmatched_ink_union_fraction",
            "contradiction_overwhelming_outside_ink_union_fraction",
        )
        for field_name in unit_interval_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")
        positive_fields = (
            "residual_changed_z_threshold",
            "occlusion_min_boundary_score",
            "occlusion_strong_boundary_score",
            "gutter_min_aspect_ratio",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

    @classmethod
    def from_json(cls, path: Path | None) -> ClusterConfig:
        """Load configuration overrides from JSON.

        Args:
            path: JSON file containing dataclass field overrides, or `None`.

        Returns:
            A validated configuration object.
        """
        if path is None:
            return cls()
        values = json.loads(path.read_text(encoding="utf-8"))
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        return cls(**values)

    def fingerprint(self) -> str:
        """Return a stable short hash of the configuration."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
