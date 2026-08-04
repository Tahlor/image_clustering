"""Regression tests for localized text-erasure occlusions and hard negatives."""

from image_clustering.clustering.candidate_scoring import pair_probabilities
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import Registration
from image_clustering.clustering.scoring_decision import _decision


def _registration(feature_overlap: float = 0.24) -> Registration:
    return Registration(
        accepted=True,
        model="affine",
        inlier_ratio=0.72,
        feature_overlap=feature_overlap,
        x_span=0.82,
        y_span=0.76,
        alignment_score=0.72,
    )


def _content(**overrides: float | int) -> ContentMetrics:
    values: dict[str, float | int] = {
        "unmatched_ink_fraction": 0.015,
        "unmatched_ink_union_fraction": 0.20,
        "ink_mismatch_tiles_fraction": 0.24,
        "coherent_ink_component_count": 4,
        "largest_ink_component_fraction": 0.006,
        "residual_tiles_changed_fraction": 0.30,
        "occlusion_candidate_count": 1,
        "occlusion_area_fraction": 0.50,
        "occlusion_residual_capture": 1.0,
        "occlusion_rectangularity": 1.0,
        "occlusion_boundary_score": 1.1,
        "occlusion_material_fraction": 0.0,
        "occlusion_material_median": 0.0,
        "outside_unmatched_ink_fraction": 0.0002,
        "outside_unmatched_ink_union_fraction": 0.0025,
        "outside_ink_mismatch_tiles_fraction": 0.018,
        "full_page_occlusion_count": 1,
        "shallow_occlusion_count": 0,
        "page_count": 2,
        "inside_unmatched_ink_union_fraction": 0.383,
        "occlusion_ink_mismatch_capture": 0.997,
        "occlusion_localization_contrast": 0.380,
    }
    values.update(overrides)
    return ContentMetrics(**values)


def test_near_background_localized_erasure_is_a_physical_occlusion() -> None:
    config = ClusterConfig()
    content = _content()

    accepted, branch, _ = _decision(
        registration=_registration(),
        change={"valid_fraction": 1.0, "changed_fraction": 0.18},
        content=content,
        config=config,
    )
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 1.0, "changed_fraction": 0.18},
        content=content,
        accepted=accepted,
        hard_contradiction=False,
        candidate_threshold=0.08,
        config=config,
    )

    assert accepted
    assert branch == "physical_occlusion"
    assert probabilities.candidate_flag


def test_page_wide_different_record_does_not_become_an_occlusion() -> None:
    config = ClusterConfig()
    content = _content(
        unmatched_ink_union_fraction=0.42,
        ink_mismatch_tiles_fraction=0.81,
        residual_tiles_changed_fraction=0.74,
        occlusion_candidate_count=2,
        occlusion_area_fraction=1.0,
        occlusion_material_fraction=0.40,
        occlusion_material_median=0.04,
        outside_unmatched_ink_fraction=0.0,
        outside_unmatched_ink_union_fraction=0.0,
        outside_ink_mismatch_tiles_fraction=0.0,
        full_page_occlusion_count=2,
        inside_unmatched_ink_union_fraction=0.42,
        occlusion_ink_mismatch_capture=1.0,
        occlusion_localization_contrast=0.42,
    )

    accepted, branch, reason = _decision(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.72},
        content=content,
        config=config,
    )

    assert not accepted
    assert branch is None
    assert "page-wide" in reason


def test_extreme_change_without_localization_contrast_is_rejected() -> None:
    config = ClusterConfig()
    content = _content(
        unmatched_ink_union_fraction=0.221,
        ink_mismatch_tiles_fraction=0.609,
        residual_tiles_changed_fraction=0.69,
        occlusion_area_fraction=0.672,
        occlusion_residual_capture=0.75,
        occlusion_rectangularity=0.72,
        occlusion_material_fraction=0.55,
        occlusion_material_median=0.033,
        outside_unmatched_ink_union_fraction=0.231,
        outside_ink_mismatch_tiles_fraction=0.393,
        full_page_occlusion_count=0,
        inside_unmatched_ink_union_fraction=0.218,
        occlusion_ink_mismatch_capture=0.75,
        occlusion_localization_contrast=0.0,
    )

    accepted, branch, _ = _decision(
        registration=_registration(feature_overlap=0.112),
        change={"valid_fraction": 0.96, "changed_fraction": 0.692},
        content=content,
        config=config,
    )

    assert not accepted
    assert branch is None
