"""Regression tests for low-contrast sheets visible mainly in the text channel."""

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics, analyze_content
from image_clustering.clustering.content_candidates import (
    _component_candidates,
    _dense_ink_candidate,
)
from image_clustering.clustering.models import Registration
from image_clustering.clustering.scoring_decision import _decision


def _tile_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.zeros((18, 24), dtype=np.float32)
    changed = np.zeros((18, 24), dtype=bool)
    valid = np.ones((18, 24), dtype=bool)
    ink = np.zeros((18, 24), dtype=bool)
    ink[5:15, 2:22] = True
    return scores, changed, valid, ink


def test_dense_ink_block_fills_only_established_rectangle() -> None:
    config = ClusterConfig()
    _, _, valid, ink = _tile_fixture()
    reference = np.full((180, 240), 238, dtype=np.uint8)
    aligned = reference.copy()

    candidate = _dense_ink_candidate(
        ink_tile_mismatch=ink,
        page_valid=valid,
        page_columns=range(24),
        page_bbox=(0, 0, 240, 180),
        shape=reference.shape,
        reference=reference,
        aligned=aligned,
        config=config,
    )

    assert candidate is not None
    support = candidate["support"]
    assert isinstance(support, np.ndarray)
    expected = np.zeros_like(valid)
    expected[5:15, 2:22] = True
    assert np.array_equal(support, expected)
    assert candidate["rectangularity"] == 1.0
    assert candidate["full_page"] == 0


def test_dense_ink_fallback_runs_only_without_residual_candidate() -> None:
    config = ClusterConfig()
    scores, changed, valid, ink = _tile_fixture()
    reference = np.full((180, 240), 238, dtype=np.uint8)
    aligned = reference.copy()

    candidates = _component_candidates(
        scores=scores,
        changed=changed,
        valid_tiles=valid,
        ink_tile_mismatch=ink,
        page_columns=range(24),
        page_bbox=(0, 0, 240, 180),
        shape=reference.shape,
        reference=reference,
        aligned=aligned,
        config=config,
    )

    assert len(candidates) == 1
    support = candidates[0]["support"]
    assert isinstance(support, np.ndarray)
    assert int(support.sum()) == 200


def test_sparse_text_disagreement_does_not_create_fallback_block() -> None:
    config = ClusterConfig()
    _, _, valid, ink = _tile_fixture()
    ink[:] = False
    ink[::3, ::4] = True
    reference = np.full((180, 240), 238, dtype=np.uint8)

    candidate = _dense_ink_candidate(
        ink_tile_mismatch=ink,
        page_valid=valid,
        page_columns=range(24),
        page_bbox=(0, 0, 240, 180),
        shape=reference.shape,
        reference=reference,
        aligned=reference.copy(),
        config=config,
    )

    assert candidate is None


def _form() -> np.ndarray:
    image = np.full((600, 420), 238, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (400, 580), 45, 2)
    cv2.putText(
        image,
        "PETITION FOR NATURALIZATION",
        (52, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        30,
        2,
        cv2.LINE_AA,
    )
    for row, y in enumerate(range(85, 555, 35)):
        cv2.line(image, (35, y), (385, y), 115, 1)
        cv2.putText(
            image,
            f"record text {row:02d}",
            (55, y - 7),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.46,
            25,
            1,
            cv2.LINE_AA,
        )
    return image


def test_near_background_sheet_is_recovered_from_text_erasure() -> None:
    config = ClusterConfig()
    reference = _form()
    aligned = reference.copy()
    cv2.rectangle(aligned, (42, 185), (378, 455), 238, -1)
    cv2.rectangle(aligned, (42, 185), (378, 455), 80, 2)
    valid = np.full(reference.shape, 255, dtype=np.uint8)

    content = analyze_content(reference, aligned, valid, config)
    accepted, branch, _ = _decision(
        registration=Registration(accepted=True, feature_overlap=0.30),
        change={"valid_fraction": 1.0, "changed_fraction": 0.45},
        content=content,
        config=config,
    )

    assert content.occlusion_candidate_count == 1
    assert content.occlusion_ink_mismatch_capture >= 0.75
    assert content.outside_unmatched_ink_union_fraction <= 0.08
    assert accepted
    assert branch == "physical_occlusion"


def test_dense_candidate_with_dirty_exterior_is_not_accepted() -> None:
    config = ClusterConfig()
    content = ContentMetrics(
        unmatched_ink_fraction=0.05,
        unmatched_ink_union_fraction=0.45,
        ink_mismatch_tiles_fraction=0.70,
        coherent_ink_component_count=20,
        largest_ink_component_fraction=0.02,
        residual_tiles_changed_fraction=0.50,
        occlusion_candidate_count=1,
        occlusion_area_fraction=0.55,
        occlusion_residual_capture=0.80,
        occlusion_rectangularity=0.75,
        occlusion_boundary_score=0.5,
        occlusion_material_fraction=0.35,
        occlusion_material_median=0.03,
        outside_unmatched_ink_fraction=0.04,
        outside_unmatched_ink_union_fraction=0.25,
        outside_ink_mismatch_tiles_fraction=0.60,
        full_page_occlusion_count=0,
        shallow_occlusion_count=0,
        page_count=1,
        inside_unmatched_ink_union_fraction=0.72,
        occlusion_ink_mismatch_capture=0.68,
        occlusion_localization_contrast=0.47,
    )

    accepted, branch, reason = _decision(
        registration=Registration(accepted=True, feature_overlap=0.30),
        change={"valid_fraction": 1.0, "changed_fraction": 0.62},
        content=content,
        config=config,
    )

    assert not accepted
    assert branch is None
    assert "page-wide" in reason or "outside" in reason or "ink" in reason
