"""Regression tests for dirty-exterior identity support in review scoring."""

import pytest

from image_clustering.clustering.candidate_scoring import (
    _dirty_exterior_identity_support,
    pair_probabilities,
)
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import Registration


def _dirty_content(**overrides: float | int) -> ContentMetrics:
    values: dict[str, float | int] = {
        "unmatched_ink_fraction": 0.116185,
        "unmatched_ink_union_fraction": 0.484218,
        "ink_mismatch_tiles_fraction": 0.628019,
        "coherent_ink_component_count": 20,
        "largest_ink_component_fraction": 0.023880,
        "residual_tiles_changed_fraction": 0.178744,
        "occlusion_candidate_count": 1,
        "occlusion_area_fraction": 0.391270,
        "occlusion_residual_capture": 0.545477,
        "occlusion_rectangularity": 0.857143,
        "occlusion_boundary_score": 0.5,
        "occlusion_material_fraction": 0.658782,
        "occlusion_material_median": 0.061897,
        "outside_unmatched_ink_fraction": 0.083773,
        "outside_unmatched_ink_union_fraction": 0.445750,
        "outside_ink_mismatch_tiles_fraction": 0.482213,
        "full_page_occlusion_count": 0,
        "shallow_occlusion_count": 0,
        "page_count": 1,
        "inside_unmatched_ink_union_fraction": 0.519272,
        "occlusion_ink_mismatch_capture": 0.561090,
        "occlusion_localization_contrast": 0.073522,
    }
    values.update(overrides)
    return ContentMetrics(**values)


def _registration(feature_overlap: float) -> Registration:
    return Registration(
        accepted=True,
        model="affine",
        inlier_ratio=0.826531,
        feature_overlap=feature_overlap,
        x_span=0.674648,
        y_span=0.627648,
        alignment_score=0.826531,
    )


def test_weak_identity_support_downweights_dirty_exterior_candidate() -> None:
    config = ClusterConfig()
    support = _dirty_exterior_identity_support(
        _registration(0.0648),
        _dirty_content(),
        config,
    )

    assert support == pytest.approx(0.053333, abs=0.00001)


def test_strong_identity_support_preserves_dirty_exterior_candidate() -> None:
    config = ClusterConfig()
    support = _dirty_exterior_identity_support(
        _registration(0.30),
        _dirty_content(),
        config,
    )

    assert support == 1.0


def test_clean_exterior_is_not_identity_gated() -> None:
    config = ClusterConfig()
    support = _dirty_exterior_identity_support(
        _registration(0.0648),
        _dirty_content(
            outside_unmatched_ink_union_fraction=0.06,
            outside_ink_mismatch_tiles_fraction=0.18,
        ),
        config,
    )

    assert support == 1.0


def test_blurred_same_template_record_drops_below_review_threshold() -> None:
    config = ClusterConfig()
    probabilities = pair_probabilities(
        registration=_registration(0.0648),
        change={"valid_fraction": 0.96, "changed_fraction": 0.711936},
        content=_dirty_content(),
        accepted=False,
        hard_contradiction=True,
        candidate_threshold=config.occlusion_candidate_probability_threshold,
        config=config,
    )

    assert probabilities.raw_occluded_given_same > 0.99
    assert probabilities.same_document > 0.10
    assert probabilities.same_occluded < 0.01
    assert not probabilities.candidate_flag


def test_noisy_true_occlusion_with_strong_registration_stays_ranked() -> None:
    config = ClusterConfig()
    probabilities = pair_probabilities(
        registration=_registration(0.30),
        change={"valid_fraction": 0.96, "changed_fraction": 0.55},
        content=_dirty_content(
            outside_unmatched_ink_union_fraction=0.33,
            outside_ink_mismatch_tiles_fraction=0.50,
            occlusion_ink_mismatch_capture=0.90,
            occlusion_localization_contrast=0.42,
            inside_unmatched_ink_union_fraction=0.75,
            occlusion_residual_capture=0.82,
        ),
        accepted=False,
        hard_contradiction=False,
        candidate_threshold=config.occlusion_candidate_probability_threshold,
        config=config,
    )

    assert probabilities.occlusion_evidence > 0.40
    assert probabilities.same_occluded >= config.occlusion_candidate_probability_threshold
    assert probabilities.candidate_flag
