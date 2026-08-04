"""Real-cohort boundaries for the conservative full-page occlusion veto."""

from image_clustering.clustering.candidate_scoring import pair_probabilities
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import Registration
from image_clustering.clustering.scoring_decision import _decision


def _registration() -> Registration:
    return Registration(
        accepted=True,
        model="affine",
        inlier_ratio=0.72,
        feature_overlap=0.24,
        x_span=0.82,
        y_span=0.76,
        alignment_score=0.72,
    )


def _content(
    *,
    material_median: float,
    inside_union: float,
    ink_tiles: float,
    material_fraction: float = 0.55,
) -> ContentMetrics:
    return ContentMetrics(
        unmatched_ink_fraction=0.05,
        unmatched_ink_union_fraction=inside_union,
        ink_mismatch_tiles_fraction=ink_tiles,
        coherent_ink_component_count=20,
        largest_ink_component_fraction=0.02,
        residual_tiles_changed_fraction=0.325,
        occlusion_candidate_count=2,
        occlusion_area_fraction=0.999,
        occlusion_residual_capture=1.0,
        occlusion_rectangularity=1.0,
        occlusion_boundary_score=0.0,
        occlusion_material_fraction=material_fraction,
        occlusion_material_median=material_median,
        outside_unmatched_ink_fraction=1.0,
        outside_unmatched_ink_union_fraction=1.0,
        outside_ink_mismatch_tiles_fraction=0.0,
        full_page_occlusion_count=2,
        shallow_occlusion_count=0,
        page_count=2,
        inside_unmatched_ink_union_fraction=inside_union,
        occlusion_ink_mismatch_capture=0.995,
        occlusion_localization_contrast=0.0,
    )


def test_photo_and_different_record_full_page_state_is_rejected() -> None:
    config = ClusterConfig()
    content = _content(
        material_median=0.0648,
        inside_union=0.493,
        ink_tiles=0.868,
    )
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.75},
        content=content,
        accepted=False,
        hard_contradiction=True,
        candidate_threshold=0.08,
        config=config,
    )
    accepted, branch, reason = _decision(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.75},
        content=content,
        config=config,
    )

    assert probabilities.occlusion_evidence < 0.05
    assert not accepted
    assert branch is None
    assert "page-wide" in reason


def test_low_contrast_full_page_sheet_remains_eligible() -> None:
    config = ClusterConfig()
    content = _content(
        material_median=0.099,
        inside_union=0.620,
        ink_tiles=0.711,
        material_fraction=0.70,
    )
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.64},
        content=content,
        accepted=True,
        hard_contradiction=False,
        candidate_threshold=0.08,
        config=config,
    )
    accepted, branch, _ = _decision(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.64},
        content=content,
        config=config,
    )

    assert probabilities.occlusion_evidence > 0.80
    assert accepted
    assert branch == "physical_occlusion"


def test_lower_material_sheet_survives_when_text_mismatch_is_not_distributed() -> None:
    config = ClusterConfig()
    content = _content(
        material_median=0.0574,
        inside_union=0.321,
        ink_tiles=0.438,
        material_fraction=0.72,
    )
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.62},
        content=content,
        accepted=True,
        hard_contradiction=False,
        candidate_threshold=0.08,
        config=config,
    )

    assert probabilities.occlusion_evidence > 0.80
