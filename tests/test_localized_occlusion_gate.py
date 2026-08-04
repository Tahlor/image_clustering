"""Regression tests for localized physical-occlusion evidence."""

import cv2
import numpy as np

from image_clustering.clustering.candidate_scoring import pair_probabilities
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics, analyze_content
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


def _content(**overrides: float | int) -> ContentMetrics:
    values: dict[str, float | int] = {
        "unmatched_ink_fraction": 0.012,
        "unmatched_ink_union_fraction": 0.08,
        "ink_mismatch_tiles_fraction": 0.12,
        "coherent_ink_component_count": 4,
        "largest_ink_component_fraction": 0.006,
        "residual_tiles_changed_fraction": 0.35,
        "occlusion_candidate_count": 1,
        "occlusion_area_fraction": 0.38,
        "occlusion_residual_capture": 0.82,
        "occlusion_rectangularity": 0.74,
        "occlusion_boundary_score": 1.2,
        "occlusion_material_fraction": 0.55,
        "occlusion_material_median": 0.045,
        "outside_unmatched_ink_fraction": 0.002,
        "outside_unmatched_ink_union_fraction": 0.018,
        "outside_ink_mismatch_tiles_fraction": 0.03,
        "full_page_occlusion_count": 0,
        "shallow_occlusion_count": 0,
        "page_count": 1,
        "inside_unmatched_ink_union_fraction": 0.72,
        "occlusion_ink_mismatch_capture": 0.84,
        "occlusion_localization_contrast": 0.68,
    }
    values.update(overrides)
    return ContentMetrics(**values)


def test_distributed_same_template_text_is_not_an_occlusion_candidate() -> None:
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.64},
        content=_content(
            unmatched_ink_union_fraction=0.29,
            ink_mismatch_tiles_fraction=0.71,
            occlusion_area_fraction=0.065,
            occlusion_residual_capture=0.19,
            inside_unmatched_ink_union_fraction=0.55,
            occlusion_ink_mismatch_capture=0.22,
            occlusion_localization_contrast=0.29,
            outside_unmatched_ink_union_fraction=0.26,
            outside_ink_mismatch_tiles_fraction=0.70,
            occlusion_rectangularity=0.52,
            occlusion_material_median=0.11,
            page_count=2,
        ),
        accepted=False,
        hard_contradiction=True,
        candidate_threshold=0.08,
    )

    assert probabilities.raw_occluded_given_same > 0.90
    assert probabilities.occlusion_evidence < 0.05
    assert probabilities.occluded_given_same < 0.05
    assert not probabilities.candidate_flag


def test_missing_contiguous_block_zeroes_occlusion_evidence() -> None:
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.55},
        content=_content(
            occlusion_candidate_count=0,
            occlusion_area_fraction=0.0,
            occlusion_residual_capture=0.0,
            inside_unmatched_ink_union_fraction=0.0,
            occlusion_ink_mismatch_capture=0.0,
            occlusion_localization_contrast=0.0,
            outside_unmatched_ink_union_fraction=0.38,
            outside_ink_mismatch_tiles_fraction=0.70,
        ),
        accepted=False,
        hard_contradiction=True,
        candidate_threshold=0.08,
    )

    assert probabilities.occlusion_evidence == 0.0
    assert probabilities.occluded_given_same == 0.0
    assert not probabilities.candidate_flag


def test_localized_block_preserves_high_occlusion_probability() -> None:
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.42},
        content=_content(),
        accepted=True,
        hard_contradiction=False,
        candidate_threshold=0.08,
    )

    assert probabilities.occlusion_evidence > 0.80
    assert probabilities.occluded_given_same > 0.50
    assert probabilities.candidate_flag
    assert probabilities.automatic_link_eligible


def test_reduced_scale_full_page_text_replacement_is_rejected() -> None:
    config = ClusterConfig()
    content = _content(
        unmatched_ink_union_fraction=0.341,
        ink_mismatch_tiles_fraction=0.735,
        residual_tiles_changed_fraction=0.219,
        occlusion_candidate_count=2,
        occlusion_area_fraction=0.998,
        occlusion_residual_capture=1.0,
        occlusion_rectangularity=1.0,
        occlusion_material_fraction=0.40,
        occlusion_material_median=0.0268,
        outside_unmatched_ink_union_fraction=0.994,
        outside_ink_mismatch_tiles_fraction=0.0,
        full_page_occlusion_count=2,
        page_count=2,
        inside_unmatched_ink_union_fraction=0.337,
        occlusion_ink_mismatch_capture=0.982,
        occlusion_localization_contrast=0.0,
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


def test_true_full_page_material_sheet_survives_text_replacement_gate() -> None:
    config = ClusterConfig()
    content = _content(
        unmatched_ink_union_fraction=0.321,
        ink_mismatch_tiles_fraction=0.438,
        residual_tiles_changed_fraction=0.623,
        occlusion_candidate_count=2,
        occlusion_area_fraction=1.0,
        occlusion_residual_capture=1.0,
        occlusion_rectangularity=1.0,
        occlusion_material_fraction=0.72,
        occlusion_material_median=0.0574,
        outside_unmatched_ink_union_fraction=0.0,
        outside_ink_mismatch_tiles_fraction=0.0,
        full_page_occlusion_count=2,
        page_count=2,
        inside_unmatched_ink_union_fraction=0.321,
        occlusion_ink_mismatch_capture=1.0,
        occlusion_localization_contrast=0.321,
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
    accepted, branch, _ = _decision(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.62},
        content=content,
        config=config,
    )

    assert probabilities.occlusion_evidence > 0.80
    assert accepted
    assert branch == "physical_occlusion"


def _form(record: str) -> np.ndarray:
    image = np.full((600, 420), 238, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (400, 580), 40, 2)
    for y in range(70, 550, 45):
        cv2.line(image, (35, y), (385, y), 100, 1)
    cv2.putText(
        image,
        "DECLARATION OF INTENTION",
        (70, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        40,
        1,
        cv2.LINE_AA,
    )
    for value, point in (
        (record, (55, 112)),
        (f"signature {record}", (85, 517)),
    ):
        cv2.putText(
            image,
            value,
            point,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.72,
            15,
            2,
            cv2.LINE_AA,
        )
    return image


def test_same_form_different_record_is_rejected_by_text_channel() -> None:
    config = ClusterConfig()
    first = _form("ALICE VERMONT 1901")
    second = _form("ROBERT CANADA 1918")
    valid = np.full(first.shape, 255, dtype=np.uint8)
    content = analyze_content(first, second, valid, config)
    accepted, branch, reason = _decision(
        registration=Registration(accepted=True, feature_overlap=0.30),
        change={"valid_fraction": 1.0, "changed_fraction": 0.20},
        content=content,
        config=config,
    )

    assert not accepted
    assert branch is None
    assert "occlusion" in reason or "ink" in reason or "form" in reason
    assert content.occlusion_candidate_count == 0 or (
        content.occlusion_ink_mismatch_capture < 0.60
    )
