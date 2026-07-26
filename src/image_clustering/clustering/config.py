"""Public clustering configuration with reviewed-data calibrated defaults."""

from __future__ import annotations

from dataclasses import dataclass

from image_clustering.clustering._config_schema import ClusterConfig as _ClusterConfig


@dataclass(frozen=True)
class ClusterConfig(_ClusterConfig):
    """Validated clustering thresholds calibrated for large physical occlusions."""

    contradiction_text_min_ink_mismatch_tiles_fraction: float = 0.09
