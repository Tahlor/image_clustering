"""Shared document-content representations for pair scoring."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig


def local_dissimilarity(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """Return illumination-tolerant local grayscale disagreement."""
    reference_float = reference.astype(np.float32) / 255.0
    aligned_float = aligned.astype(np.float32) / 255.0
    kernel = (11, 11)
    reference_mean = cv2.GaussianBlur(reference_float, kernel, 0)
    aligned_mean = cv2.GaussianBlur(aligned_float, kernel, 0)
    reference_sq_mean = cv2.GaussianBlur(reference_float**2, kernel, 0)
    aligned_sq_mean = cv2.GaussianBlur(aligned_float**2, kernel, 0)
    cross_mean = cv2.GaussianBlur(reference_float * aligned_float, kernel, 0)
    reference_var = np.maximum(reference_sq_mean - reference_mean**2, 0)
    aligned_var = np.maximum(aligned_sq_mean - aligned_mean**2, 0)
    covariance = cross_mean - reference_mean * aligned_mean
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * reference_mean * aligned_mean + c1) * (2 * covariance + c2)
    denominator = (reference_mean**2 + aligned_mean**2 + c1) * (
        reference_var + aligned_var + c2
    )
    ssim = np.clip(numerator / np.maximum(denominator, 1e-6), -1.0, 1.0)
    absolute_difference = cv2.GaussianBlur(
        np.abs(reference_float - aligned_float),
        (9, 9),
        0,
    )
    return np.clip(0.55 * absolute_difference + 0.45 * (1.0 - ssim), 0, 1)


def _ink_mask(
    image: np.ndarray,
    core: np.ndarray,
    config: ClusterConfig,
) -> np.ndarray:
    sigma = max(5.0, min(image.shape) * config.ink_background_sigma_fraction)
    values = image.astype(np.float32)
    background = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma)
    dark = np.clip((background - values) / 64.0, 0.0, 1.0)
    gradient_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    valid_gradient = gradient[core]
    if valid_gradient.size:
        low, high = np.percentile(valid_gradient, [50, 95])
        gradient = np.clip((gradient - low) / max(high - low, 1.0), 0.0, 1.0)
    else:
        gradient = np.zeros_like(values)
    response = np.clip(dark + config.ink_gradient_weight * gradient, 0.0, 1.0)
    valid_response = np.clip(response[core] * 255.0, 0, 255).astype(np.uint8)
    if valid_response.size < 100:
        return np.zeros_like(core)
    threshold, _ = cv2.threshold(
        valid_response.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold_fraction = max(
        config.ink_min_response,
        float(threshold) / 255.0,
    )
    mask = (response >= threshold_fraction) & core
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros_like(mask)
    minimum_area = max(
        3,
        round(mask.size * config.ink_min_component_fraction),
    )
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            cleaned[labels == label] = True
    return cleaned


def _clean_mismatch(
    mismatch: np.ndarray,
    config: ClusterConfig,
) -> tuple[np.ndarray, list[int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mismatch.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros_like(mismatch)
    minimum_area = max(
        4,
        round(mismatch.size * config.ink_mismatch_min_component_fraction),
    )
    areas: list[int] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            cleaned[labels == label] = True
            areas.append(area)
    return cleaned, areas
