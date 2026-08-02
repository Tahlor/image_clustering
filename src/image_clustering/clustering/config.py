"""Public clustering configuration with reviewed-data calibrated defaults."""

from __future__ import annotations

from dataclasses import dataclass

from image_clustering.clustering._config_schema import ClusterConfig as _ClusterConfig


@dataclass(frozen=True)
class ClusterConfig(_ClusterConfig):
    """Validated clustering thresholds calibrated for large physical occlusions."""

    contradiction_text_min_ink_mismatch_tiles_fraction: float = 0.09

    # SIFT remains the primary registration. This fallback is deliberately limited
    # to same-orientation, small-motion captures and runs only after SIFT fails.
    ecc_fallback_enabled: bool = True
    ecc_min_correlation: float = 0.30
    ecc_max_rotation_degrees: float = 5.0
    ecc_max_translation_fraction: float = 0.18
    ecc_max_iterations: int = 100
    ecc_epsilon: float = 0.000001
    ecc_gaussian_filter_size: int = 5

    # Avoid full ECC on obviously unrelated pages. A fallback attempt requires a
    # bounded small-motion affine seed, enough exact descriptor matches, or coarse
    # phase-correlation evidence. Content scoring remains at full resolution.
    ecc_coarse_dimension: int = 192
    ecc_min_phase_correlation: float = 0.12
    ecc_min_descriptor_matches: int = 50

    # The continuous score is a recall-oriented review signal. It never bypasses
    # deterministic acceptance or hard-contradiction graph safeguards.
    occlusion_candidate_probability_threshold: float = 0.08

    # Extreme material changes may hide most content. When the remaining exterior
    # is dirty, require stronger identity support before creating an automatic edge.
    occlusion_dirty_exterior_min_feature_overlap: float = 0.15
    occlusion_dirty_exterior_min_alignment_score: float = 0.55
    occlusion_dirty_exterior_min_unmatched_ink_union_fraction: float = 0.10
    occlusion_dirty_exterior_min_ink_mismatch_tiles_fraction: float = 0.40

    # Automatic graph edges need plausible capture-sequence proximity. Pairs beyond
    # this filename suffix gap are retained as review candidates, not auto-linked.
    automatic_link_max_numeric_filename_gap: int = 12
    automatic_link_require_same_filename_prefix: bool = True
    automatic_link_allow_full_page_ecc: bool = False

    def __post_init__(self) -> None:
        """Validate base thresholds and recall-first registration settings."""
        super().__post_init__()
        for name in (
            "ecc_min_correlation",
            "ecc_max_translation_fraction",
            "ecc_min_phase_correlation",
            "occlusion_candidate_probability_threshold",
            "occlusion_dirty_exterior_min_feature_overlap",
            "occlusion_dirty_exterior_min_alignment_score",
            "occlusion_dirty_exterior_min_unmatched_ink_union_fraction",
            "occlusion_dirty_exterior_min_ink_mismatch_tiles_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.ecc_max_rotation_degrees <= 0:
            raise ValueError("ecc_max_rotation_degrees must be positive")
        if self.ecc_max_iterations < 1:
            raise ValueError("ecc_max_iterations must be positive")
        if self.ecc_epsilon <= 0:
            raise ValueError("ecc_epsilon must be positive")
        if self.ecc_coarse_dimension < 64:
            raise ValueError("ecc_coarse_dimension must be at least 64")
        if self.ecc_min_descriptor_matches < 0:
            raise ValueError("ecc_min_descriptor_matches cannot be negative")
        if self.automatic_link_max_numeric_filename_gap < 1:
            raise ValueError(
                "automatic_link_max_numeric_filename_gap must be positive"
            )
        if (
            self.ecc_gaussian_filter_size < 1
            or self.ecc_gaussian_filter_size % 2 == 0
        ):
            raise ValueError("ecc_gaussian_filter_size must be a positive odd integer")
