"""Focused regressions for review-only material-occlusion recovery."""

from image_clustering.clustering import candidate_scoring
from image_clustering.clustering.candidate_scoring import (
    _low_text_material_candidate,
    _strong_material_block_candidate,
    pair_probabilities,
)
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import Registration


def _content(**overrides: float | int) -> ContentMetrics:
    values: dict[str, float | int] = {
        "unmatched_ink_fraction": 0.02,
        "unmatched_ink_union_fraction": 0.20,
        "ink_mismatch_tiles_fraction": 0.30,
        "coherent_ink_component_count": 1,
        "largest_ink_component_fraction": 0.20,
        "residual_tiles_changed_fraction": 0.45,
        "occlusion_candidate_count": 1,
        "occlusion_area_fraction": 0.35,
        "occlusion_residual_capture": 0.90,
        "occlusion_rectangularity": 0.80,
        "occlusion_boundary_score": 1.0,
        "occlusion_material_fraction": 0.60,
        "occlusion_material_median": 0.04,
        "outside_unmatched_ink_fraction": 0.002,
        "outside_unmatched_ink_union_fraction": 0.01,
        "outside_ink_mismatch_tiles_fraction": 0.04,
        "full_page_occlusion_count": 0,
        "shallow_occlusion_count": 0,
        "page_count": 1,
        "inside_unmatched_ink_union_fraction": 0.40,
        "occlusion_ink_mismatch_capture": 0.95,
        "occlusion_localization_contrast": 0.30,
    }
    values.update(overrides)
    return ContentMetrics(**values)


def test_review_candidate_threshold_is_recall_first() -> None:
    assert ClusterConfig().occlusion_candidate_probability_threshold == 0.04


def test_strong_material_block_surfaces_thin_paper_strip() -> None:
    assert _strong_material_block_candidate(
        p_same=0.95,
        content=_content(),
        occlusion_evidence=0.85,
    )


def test_strong_material_block_rejects_dirty_exterior() -> None:
    assert not _strong_material_block_candidate(
        p_same=0.95,
        content=_content(outside_unmatched_ink_union_fraction=0.08),
        occlusion_evidence=0.85,
    )


def test_low_text_material_state_requires_sift_not_ecc() -> None:
    content = _content(
        inside_unmatched_ink_union_fraction=0.02,
        occlusion_localization_contrast=0.02,
    )
    sift = Registration(
        accepted=True,
        model="affine",
        feature_overlap=0.50,
        fallback_used=False,
    )
    ecc = Registration(
        accepted=True,
        model="ecc_euclidean",
        feature_overlap=0.50,
        fallback_used=True,
    )

    assert _low_text_material_candidate(0.98, sift, content)
    assert not _low_text_material_candidate(0.98, ecc, content)


def test_material_override_is_review_only(
    monkeypatch,
) -> None:
    probabilities = iter((0.99, 0.001))
    monkeypatch.setattr(
        candidate_scoring,
        "_linear_probability",
        lambda intercept, coefficients, values: next(probabilities),
    )
    monkeypatch.setattr(
        candidate_scoring,
        "_occlusion_evidence",
        lambda content, config: 0.90,
    )
    monkeypatch.setattr(
        candidate_scoring,
        "_dirty_exterior_identity_support",
        lambda registration, content, config: 1.0,
    )
    monkeypatch.setattr(
        candidate_scoring,
        "_localized_text_erasure",
        lambda content, config: False,
    )

    result = pair_probabilities(
        registration=Registration(
            accepted=True,
            model="affine",
            feature_overlap=0.50,
            fallback_used=False,
        ),
        change={"valid_fraction": 0.98, "changed_fraction": 0.40},
        content=_content(),
        accepted=False,
        hard_contradiction=False,
        candidate_threshold=1.0,
        config=ClusterConfig(),
    )

    assert result.candidate_flag
    assert not result.automatic_link_eligible
    assert result.model_version == "vermont-synthetic-logit-v6-material-block-recall"
