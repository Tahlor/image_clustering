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
        )
        for field_name in unit_interval_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")

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
