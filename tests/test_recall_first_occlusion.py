"""Tests for recall-first occlusion registration and scoring."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from image_clustering.clustering.candidate_scoring import pair_probabilities
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import (
    ImageFeatures,
    ImageItem,
    PairComparison,
    Registration,
)
from image_clustering.clustering.registration import register_pair


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
    }
    values.update(overrides)
    return ContentMetrics(**values)


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


def test_probability_states_are_coherent() -> None:
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.42},
        content=_content(),
        accepted=True,
        hard_contradiction=False,
        candidate_threshold=ClusterConfig().occlusion_candidate_probability_threshold,
    )

    assert 0.0 <= probabilities.same_document <= 1.0
    assert 0.0 <= probabilities.occluded_given_same <= 1.0
    assert probabilities.same_clean + probabilities.same_occluded == pytest.approx(
        probabilities.same_document
    )
    assert probabilities.same_clean + probabilities.same_occluded + (
        probabilities.different_document
    ) == pytest.approx(1.0)


def test_probability_never_overrides_hard_contradiction() -> None:
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.42},
        content=_content(),
        accepted=True,
        hard_contradiction=True,
        candidate_threshold=0.0,
    )

    assert probabilities.candidate_flag
    assert not probabilities.automatic_link_eligible


def test_rejected_pair_can_be_review_candidate_but_not_link() -> None:
    probabilities = pair_probabilities(
        registration=_registration(),
        change={"valid_fraction": 0.96, "changed_fraction": 0.42},
        content=_content(),
        accepted=False,
        hard_contradiction=False,
        candidate_threshold=0.0,
    )

    assert probabilities.candidate_flag
    assert not probabilities.automatic_link_eligible


def test_probability_diagnostics_round_trip() -> None:
    comparison = PairComparison(
        first_image_id="a.jpg",
        second_image_id="b.jpg",
        sequence_id="sequence",
        index_gap=1,
        same_document=False,
        confidence=0.49,
        reason="review candidate",
        registration_fallback_used=True,
        registration_alignment_score=0.81,
        probability_model_version="vermont-synthetic-logit-v1",
        same_document_probability=0.82,
        occluded_given_same_probability=0.74,
        same_clean_probability=0.2132,
        same_occluded_probability=0.6068,
        different_document_probability=0.18,
        occlusion_candidate_flag=True,
        automatic_link_eligible=False,
    )

    assert PairComparison.from_dict(comparison.to_dict()) == comparison


def test_old_pair_payload_uses_safe_probability_defaults() -> None:
    restored = PairComparison.from_dict(
        {
            "first_image_id": "a.jpg",
            "second_image_id": "b.jpg",
            "sequence_id": "sequence",
            "index_gap": 1,
            "same_document": True,
            "confidence": 0.9,
            "reason": "legacy",
        }
    )

    assert restored.same_document
    assert restored.same_document_probability == 0.0
    assert restored.different_document_probability == 1.0
    assert not restored.occlusion_candidate_flag


def _document() -> np.ndarray:
    image = np.full((640, 460), 238, dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (442, 622), 40, 2)
    cv2.putText(
        image,
        "PETITION FOR NATURALIZATION",
        (50, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        25,
        2,
        cv2.LINE_AA,
    )
    for y in range(95, 590, 42):
        cv2.line(image, (35, y), (425, y), 105, 1)
    for x in (145, 285):
        cv2.line(image, (x, 95), (x, 555), 145, 1)
    for text, point in (
        ("MARY JOHNSON", (55, 132)),
        ("BURLINGTON VERMONT", (62, 258)),
        ("1924", (315, 342)),
        ("signature Mary Johnson", (95, 575)),
    ):
        cv2.putText(
            image,
            text,
            point,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.68,
            20,
            2,
            cv2.LINE_AA,
        )
    return image


def _features(image: np.ndarray, name: str) -> ImageFeatures:
    detector = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.025)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return ImageFeatures(
        image=ImageItem(name, Path(name), "sequence", 0),
        gray=image,
        scale=1.0,
        keypoints_xy=np.float32([point.pt for point in keypoints]),
        descriptors=(
            descriptors
            if descriptors is not None
            else np.empty((0, 128), dtype=np.float32)
        ),
    )


def test_small_motion_fallback_recovers_large_occlusion() -> None:
    previous = _document()
    center = (previous.shape[1] / 2, previous.shape[0] / 2)
    motion = cv2.getRotationMatrix2D(center, 1.1, 1.0)
    motion[:, 2] += np.array([7.0, -5.0])
    current = cv2.warpAffine(
        previous,
        motion,
        (previous.shape[1], previous.shape[0]),
        borderValue=238,
    )
    cv2.rectangle(current, (35, 215), (425, 505), 247, -1)
    cv2.rectangle(current, (35, 215), (425, 505), 75, 2)
    cv2.putText(
        current,
        "CERTIFICATE OF ARRIVAL",
        (65, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        30,
        2,
        cv2.LINE_AA,
    )

    config = ClusterConfig(min_inliers=1000, max_features=2500)
    registration = register_pair(
        previous=_features(previous, "previous.jpg"),
        current=_features(current, "current.jpg"),
        config=config,
    )

    assert registration.accepted
    assert registration.model == "ecc_euclidean"
    assert registration.fallback_used
    assert registration.alignment_score >= config.ecc_min_correlation


def test_small_motion_fallback_rejects_incompatible_orientation() -> None:
    previous = _document()
    current = cv2.rotate(previous, cv2.ROTATE_90_CLOCKWISE)
    config = ClusterConfig(min_inliers=1000, max_features=2500)

    registration = register_pair(
        previous=_features(previous, "previous.jpg"),
        current=_features(current, "current.jpg"),
        config=config,
    )

    assert not registration.accepted
    assert registration.fallback_used
