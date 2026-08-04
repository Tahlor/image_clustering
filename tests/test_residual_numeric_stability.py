"""Regression tests for residual and exposure normalization stability."""

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_features import local_dissimilarity
from image_clustering.clustering.scoring import _normalize_brightness


def _document() -> np.ndarray:
    image = np.full((480, 720), 232, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (719, 479), 0, 24)
    for y in range(70, 430, 45):
        cv2.line(image, (55, y), (665, y), 105, 1)
    for row, text in enumerate(("NAME ALICE", "BIRTH 1901", "SIGNATURE")):
        cv2.putText(
            image,
            text,
            (85, 115 + row * 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            35,
            2,
            cv2.LINE_AA,
        )
    return image


def test_identical_dark_regions_have_zero_local_dissimilarity() -> None:
    image = _document()

    dissimilarity = local_dissimilarity(image, image)

    assert float(dissimilarity.max()) < 1e-10


def test_large_cover_does_not_bias_exterior_exposure_fit() -> None:
    config = ClusterConfig()
    reference = _document()
    aligned = np.clip(reference.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    cv2.rectangle(aligned, (390, 90), (685, 390), 35, -1)
    valid = np.full(reference.shape, 255, dtype=np.uint8)

    normalized = _normalize_brightness(reference, aligned, valid, config)

    outside = np.ones(reference.shape, dtype=bool)
    outside[80:401, 380:696] = False
    outside[:28] = False
    outside[-28:] = False
    outside[:, :28] = False
    outside[:, -28:] = False
    exterior_error = np.abs(
        reference[outside].astype(np.int16) - normalized[outside].astype(np.int16)
    )
    assert float(np.median(exterior_error)) <= 1.0
    assert float(np.percentile(exterior_error, 95)) <= 3.0
