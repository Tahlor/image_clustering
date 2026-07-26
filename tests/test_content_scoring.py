"""Regression tests for document-specific ink and occlusion scoring."""

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import (
    ContentMetrics,
    analyze_content,
    local_dissimilarity,
)
from image_clustering.clustering.models import Registration
from image_clustering.clustering.scoring_decision import (
    _decision,
    _hard_contradiction,
)


def _form_page(name: str = "ALICE", date: str = "1901") -> np.ndarray:
    image = np.full((600, 420), 238, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (400, 580), 40, 2)
    for y in range(70, 550, 45):
        cv2.line(image, (35, y), (385, y), 100, 1)
    for x in (140, 270):
        cv2.line(image, (x, 70), (x, 520), 150, 1)
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
    if name:
        cv2.putText(
            image,
            name,
            (55, 112),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.8,
            20,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"signature {name}",
            (90, 540),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.7,
            20,
            2,
            cv2.LINE_AA,
        )
    if date:
        cv2.putText(
            image,
            date,
            (285, 157),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.7,
            20,
            2,
            cv2.LINE_AA,
        )
    return image


def _filled_form(record: int) -> np.ndarray:
    """Return the same printed form populated with a different record."""
    image = _form_page(name="", date="")
    values = (
        ["ALICE", "VERMONT", "1901", "FARMER", "BURLINGTON", "ARCHIBALD"]
        if record == 1
        else ["ROBERT", "CANADA", "1918", "CARPENTER", "MONTPELIER", "SMITH"]
    )
    positions = [(48, 112), (150, 157), (285, 157), (55, 247), (150, 337), (85, 517)]
    for value, position in zip(values, positions, strict=True):
        cv2.putText(
            image,
            value,
            position,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.72,
            15,
            2,
            cv2.LINE_AA,
        )
    return image


def _decision_for(
    reference: np.ndarray,
    aligned: np.ndarray,
) -> tuple[bool, str | None, str, ContentMetrics]:
    config = ClusterConfig()
    valid = np.full(reference.shape, 255, dtype=np.uint8)
    core = cv2.erode(valid, np.ones((9, 9), np.uint8)) > 0
    dissimilarity = local_dissimilarity(reference, aligned)
    changed_fraction = float(
        ((dissimilarity > config.change_threshold) & core).sum() / max(core.sum(), 1)
    )
    content = analyze_content(reference, aligned, valid, config)
    accepted, branch, reason = _decision(
        registration=Registration(accepted=True, feature_overlap=0.30),
        change={"valid_fraction": 1.0, "changed_fraction": changed_fraction},
        content=content,
        config=config,
    )
    return accepted, branch, reason, content


def test_exposure_shift_is_near_duplicate() -> None:
    reference = _form_page()
    shifted = np.clip(reference.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    accepted, branch, _, _ = _decision_for(reference, shifted)
    assert (accepted, branch) == (True, "near_duplicate")


def test_same_template_with_different_handwriting_is_rejected() -> None:
    reference = _filled_form(1)
    different_record = _filled_form(2)
    accepted, branch, reason, content = _decision_for(reference, different_record)
    assert not accepted
    assert branch is None
    assert "ink" in reason or "form" in reason
    assert _hard_contradiction(False, content, ClusterConfig())


def test_real_rectangular_overlay_is_accepted() -> None:
    reference = _form_page()
    overlay = reference.copy()
    cv2.rectangle(overlay, (35, 130), (385, 360), 245, -1)
    cv2.rectangle(overlay, (35, 130), (385, 360), 80, 2)
    cv2.putText(
        overlay,
        "CERTIFICATE OF ARRIVAL",
        (75, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        30,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "JOHN SMITH",
        (90, 230),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        0.9,
        20,
        2,
        cv2.LINE_AA,
    )
    accepted, branch, _, _ = _decision_for(reference, overlay)
    assert (accepted, branch) == (True, "physical_occlusion")


def test_large_skewed_polygon_overlay_is_accepted() -> None:
    reference = _form_page()
    overlay = reference.copy()
    polygon = np.array([(32, 155), (382, 125), (394, 405), (58, 430)], np.int32)
    cv2.fillConvexPoly(overlay, polygon, 246)
    cv2.polylines(overlay, [polygon], True, 70, 3, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "PETITION FOR NATURALIZATION",
        (52, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        25,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "MARY JOHNSON 1924",
        (70, 300),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        0.78,
        20,
        2,
        cv2.LINE_AA,
    )
    accepted, branch, _, content = _decision_for(reference, overlay)
    assert (accepted, branch) == (True, "physical_occlusion")
    assert content.occlusion_area_fraction >= 0.30


def test_plausible_multi_occlusion_rejection_is_not_hard_contradiction() -> None:
    content = ContentMetrics(
        unmatched_ink_fraction=0.04,
        unmatched_ink_union_fraction=0.30,
        ink_mismatch_tiles_fraction=0.55,
        coherent_ink_component_count=6,
        largest_ink_component_fraction=0.02,
        residual_tiles_changed_fraction=0.20,
        occlusion_candidate_count=2,
        occlusion_area_fraction=0.80,
        occlusion_residual_capture=0.75,
        occlusion_rectangularity=0.80,
        occlusion_boundary_score=1.7,
        occlusion_material_fraction=0.70,
        occlusion_material_median=0.25,
        outside_unmatched_ink_fraction=0.01,
        outside_unmatched_ink_union_fraction=0.03,
        outside_ink_mismatch_tiles_fraction=0.08,
        full_page_occlusion_count=1,
        shallow_occlusion_count=1,
        page_count=2,
    )
    assert not _hard_contradiction(False, content, ClusterConfig())


def test_distributed_exterior_noise_is_hard_contradiction() -> None:
    content = ContentMetrics(
        unmatched_ink_fraction=0.03,
        unmatched_ink_union_fraction=0.04,
        ink_mismatch_tiles_fraction=0.16,
        coherent_ink_component_count=12,
        largest_ink_component_fraction=0.01,
        residual_tiles_changed_fraction=0.52,
        occlusion_candidate_count=2,
        occlusion_area_fraction=0.60,
        occlusion_residual_capture=0.90,
        occlusion_rectangularity=0.60,
        occlusion_boundary_score=0.7,
        occlusion_material_fraction=0.34,
        occlusion_material_median=0.01,
        outside_unmatched_ink_fraction=0.02,
        outside_unmatched_ink_union_fraction=0.13,
        outside_ink_mismatch_tiles_fraction=0.06,
        full_page_occlusion_count=0,
        shallow_occlusion_count=0,
        page_count=2,
    )
    assert _hard_contradiction(False, content, ClusterConfig())


def test_overwhelming_distributed_ink_is_hard_contradiction() -> None:
    content = ContentMetrics(
        unmatched_ink_fraction=0.08,
        unmatched_ink_union_fraction=0.46,
        ink_mismatch_tiles_fraction=0.90,
        coherent_ink_component_count=20,
        largest_ink_component_fraction=0.01,
        residual_tiles_changed_fraction=0.07,
        occlusion_candidate_count=1,
        occlusion_area_fraction=0.25,
        occlusion_residual_capture=0.40,
        occlusion_rectangularity=0.55,
        occlusion_boundary_score=0.8,
        occlusion_material_fraction=0.39,
        occlusion_material_median=0.10,
        outside_unmatched_ink_fraction=0.06,
        outside_unmatched_ink_union_fraction=0.36,
        outside_ink_mismatch_tiles_fraction=0.82,
        full_page_occlusion_count=0,
        shallow_occlusion_count=0,
        page_count=2,
    )
    assert _hard_contradiction(False, content, ClusterConfig())
