"""Regression tests for document-specific ink and occlusion scoring."""

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics, analyze_content
from image_clustering.clustering.models import Registration
from image_clustering.clustering.scoring import _decision, _hard_contradiction


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
        date,
        (285, 157),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        0.7,
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
    return image


def _decision_for(
    reference: np.ndarray, aligned: np.ndarray
) -> tuple[bool, str | None]:
    config = ClusterConfig()
    valid = np.full(reference.shape, 255, dtype=np.uint8)
    content = analyze_content(reference, aligned, valid, config)
    accepted, branch, _ = _decision(
        registration=Registration(accepted=True, feature_overlap=0.30),
        change={"valid_fraction": 1.0, "changed_fraction": 0.10},
        content=content,
        config=config,
    )
    return accepted, branch


def test_exposure_shift_is_near_duplicate() -> None:
    reference = _form_page()
    shifted = np.clip(reference.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    assert _decision_for(reference, shifted) == (True, "near_duplicate")


def test_same_template_with_different_handwriting_is_rejected() -> None:
    reference = _form_page("ALICE", "1901")
    different_record = _form_page("ROBERT", "1918")
    accepted, branch = _decision_for(reference, different_record)
    assert not accepted
    assert branch is None


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
    assert _decision_for(reference, overlay) == (True, "physical_occlusion")


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
