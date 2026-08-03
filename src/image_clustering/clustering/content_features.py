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


def _ink_response(
    image: np.ndarray,
    core: np.ndarray,
    config: ClusterConfig,
) -> np.ndarray:
    """Return a smooth stroke-likelihood image for handwriting and print."""
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
    return np.clip(dark + config.ink_gradient_weight * gradient, 0.0, 1.0)


def _shared_ink_threshold(
    responses: tuple[np.ndarray, ...],
    core: np.ndarray,
    config: ClusterConfig,
) -> float:
    """Choose one threshold for both registered views.

    Independent Otsu thresholds make the darker scan look as though it contains
    document-specific ink that is absent from the lighter scan. A pooled threshold
    preserves the relative stroke evidence and makes the text comparison symmetric.
    """
    valid = [
        np.clip(response[core] * 255.0, 0, 255).astype(np.uint8)
        for response in responses
    ]
    valid = [values for values in valid if values.size]
    if not valid or sum(values.size for values in valid) < 100:
        return 1.0
    pooled = np.concatenate(valid)
    threshold, _ = cv2.threshold(
        pooled.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return max(config.ink_min_response, float(threshold) / 255.0)


def _clean_ink_mask(
    response: np.ndarray,
    core: np.ndarray,
    threshold_fraction: float,
    config: ClusterConfig,
) -> np.ndarray:
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


def _ink_mask(
    image: np.ndarray,
    core: np.ndarray,
    config: ClusterConfig,
) -> np.ndarray:
    """Return a single-image stroke mask for compatibility and diagnostics."""
    response = _ink_response(image, core, config)
    threshold = _shared_ink_threshold((response,), core, config)
    return _clean_ink_mask(response, core, threshold, config)


def _paired_ink_masks(
    reference: np.ndarray,
    aligned: np.ndarray,
    core: np.ndarray,
    config: ClusterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return registered text-channel masks using one adaptive threshold."""
    reference_response = _ink_response(reference, core, config)
    aligned_response = _ink_response(aligned, core, config)
    threshold = _shared_ink_threshold(
        (reference_response, aligned_response),
        core,
        config,
    )
    return (
        _clean_ink_mask(reference_response, core, threshold, config),
        _clean_ink_mask(aligned_response, core, threshold, config),
    )


def _clean_mismatch(
    mismatch: np.ndarray,
    config: ClusterConfig,
) -> tuple[np.ndarray, list[int]]:
    """Remove isolated scan noise while retaining connected text strokes.

    Components are found on a narrowly bridged mask so anti-aliased fragments of a
    letter remain one object. Only original mismatch pixels are retained; the bridge
    cannot manufacture a large occlusion block or connect distant filled fields.
    """
    bridge_radius = max(1, round(min(mismatch.shape) * 0.0015))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * bridge_radius + 1, 2 * bridge_radius + 1),
    )
    bridged = cv2.morphologyEx(
        mismatch.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    )
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        bridged,
        connectivity=8,
    )
    cleaned = np.zeros_like(mismatch)
    minimum_area = max(
        4,
        round(mismatch.size * config.ink_mismatch_min_component_fraction),
    )
    areas: list[int] = []
    for label in range(1, count):
        component = labels == label
        original_area = int((mismatch & component).sum())
        if original_area >= minimum_area:
            cleaned[mismatch & component] = True
            areas.append(original_area)
    return cleaned, areas
