"""Pairwise same-physical-document scoring."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.models import (
    ImageFeatures,
    PairComparison,
    Registration,
)
from image_clustering.clustering.registration import (
    register_pair,
    source_pixel_transform,
    warp_current,
)


def _normalize_brightness(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    core = cv2.erode(valid_mask, np.ones((9, 9), np.uint8)) > 0
    if int(core.sum()) < 1000:
        return aligned
    reference_values = reference[core].astype(np.float32)
    aligned_values = aligned[core].astype(np.float32)
    reference_median = float(np.median(reference_values))
    aligned_median = float(np.median(aligned_values))
    reference_mad = float(np.median(np.abs(reference_values - reference_median)))
    aligned_mad = float(np.median(np.abs(aligned_values - aligned_median)))
    scale = float(np.clip(reference_mad / max(aligned_mad, 1.0), 0.75, 1.35))
    normalized = aligned.astype(np.float32) * scale
    normalized += reference_median - scale * aligned_median
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _local_dissimilarity(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
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
    numerator = (2 * reference_mean * aligned_mean + c1) * (
        2 * covariance + c2
    )
    denominator = (
        reference_mean**2 + aligned_mean**2 + c1
    ) * (reference_var + aligned_var + c2)
    ssim = np.clip(numerator / np.maximum(denominator, 1e-6), -1.0, 1.0)
    absolute_difference = cv2.GaussianBlur(
        np.abs(reference_float - aligned_float),
        (9, 9),
        0,
    )
    return np.clip(0.55 * absolute_difference + 0.45 * (1.0 - ssim), 0, 1)


def _change_metrics(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    config: ClusterConfig,
) -> dict[str, float]:
    core = cv2.erode(valid_mask, np.ones((9, 9), np.uint8)) > 0
    valid_fraction = float(core.mean())
    if int(core.sum()) < 1000:
        return {
            "valid_fraction": valid_fraction,
            "changed_fraction": 1.0,
            "stable_fraction": 0.0,
            "tiles_changed_fraction": 1.0,
            "largest_change_share": 0.0,
        }
    dissimilarity = _local_dissimilarity(reference=reference, aligned=aligned)
    raw_changed = (dissimilarity > config.change_threshold) & core
    changed_fraction = float(raw_changed.sum() / max(core.sum(), 1))
    stable_fraction = 1.0 - changed_fraction

    tile_changed = []
    height, width = reference.shape
    for row in range(config.tile_rows):
        for column in range(config.tile_columns):
            y0 = round(row * height / config.tile_rows)
            y1 = round((row + 1) * height / config.tile_rows)
            x0 = round(column * width / config.tile_columns)
            x1 = round((column + 1) * width / config.tile_columns)
            tile_core = core[y0:y1, x0:x1]
            if tile_core.mean() < 0.5:
                continue
            tile_fraction = float(raw_changed[y0:y1, x0:x1][tile_core].mean())
            tile_changed.append(tile_fraction > config.tile_changed_threshold)
    tiles_changed_fraction = float(np.mean(tile_changed)) if tile_changed else 1.0

    mask = raw_changed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    close_size = max(7, (round(min(reference.shape) * 0.018) // 2) * 2 + 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    coherent_area = sum(areas)
    largest_change_share = max(areas, default=0) / max(coherent_area, 1)
    return {
        "valid_fraction": valid_fraction,
        "changed_fraction": changed_fraction,
        "stable_fraction": stable_fraction,
        "tiles_changed_fraction": tiles_changed_fraction,
        "largest_change_share": float(largest_change_share),
    }


def _decision(
    registration: Registration,
    metrics: dict[str, float],
    config: ClusterConfig,
) -> tuple[bool, str | None, str]:
    if metrics["valid_fraction"] < config.min_valid_fraction:
        return False, None, "insufficient valid overlap after registration"
    feature_overlap = registration.feature_overlap
    changed_fraction = metrics["changed_fraction"]
    if (
        feature_overlap >= config.exceptional_min_feature_overlap
        and changed_fraction <= config.exceptional_max_changed_fraction
    ):
        return True, "exceptional_support", "very strong exact feature support"
    if (
        feature_overlap >= config.heavy_min_feature_overlap
        and changed_fraction <= config.heavy_max_changed_fraction
    ):
        return True, "heavy_occlusion", "strong exact support despite heavy occlusion"
    if (
        feature_overlap >= config.standard_min_feature_overlap
        and changed_fraction <= config.standard_max_changed_fraction
    ):
        return True, "standard", "exact support and bounded changed area"
    if feature_overlap < config.standard_min_feature_overlap:
        return False, None, "too little document-specific exact feature support"
    if changed_fraction > config.standard_max_changed_fraction:
        return False, None, "differences are too page-wide for available exact support"
    return False, None, "same-document validation failed"


def score_pair(
    previous: ImageFeatures,
    current: ImageFeatures,
    index_gap: int,
    config: ClusterConfig,
) -> PairComparison:
    """Score whether two captures show the same physical document scene."""
    registration = register_pair(previous=previous, current=current, config=config)
    if not registration.accepted:
        return PairComparison(
            first_image_id=previous.image.image_id,
            second_image_id=current.image.image_id,
            sequence_id=previous.image.sequence_id,
            index_gap=index_gap,
            same_document=False,
            confidence=0.0,
            reason=registration.reason or "registration rejected",
            good_match_count=registration.good_match_count,
        )
    aligned, valid_mask = warp_current(
        current_gray=current.gray,
        registration=registration,
        previous_shape=previous.gray.shape,
    )
    aligned = _normalize_brightness(
        reference=previous.gray,
        aligned=aligned,
        valid_mask=valid_mask,
    )
    metrics = _change_metrics(
        reference=previous.gray,
        aligned=aligned,
        valid_mask=valid_mask,
        config=config,
    )
    accepted, branch, reason = _decision(
        registration=registration,
        metrics=metrics,
        config=config,
    )
    support_score = np.clip(
        (registration.feature_overlap - 0.06) / 0.24,
        0.0,
        1.0,
    )
    bounded_change_score = np.clip(
        (0.90 - metrics["changed_fraction"]) / 0.55,
        0.0,
        1.0,
    )
    confidence = float(0.75 * support_score + 0.25 * bounded_change_score)
    confidence = max(confidence, 0.50) if accepted else min(confidence, 0.49)
    return PairComparison(
        first_image_id=previous.image.image_id,
        second_image_id=current.image.image_id,
        sequence_id=previous.image.sequence_id,
        index_gap=index_gap,
        same_document=accepted,
        confidence=confidence,
        reason=reason,
        registration_model=registration.model,
        transform=source_pixel_transform(
            registration=registration,
            previous_scale=previous.scale,
            current_scale=current.scale,
        ),
        good_match_count=registration.good_match_count,
        inlier_count=registration.inlier_count,
        inlier_ratio=registration.inlier_ratio,
        feature_overlap=registration.feature_overlap,
        median_reprojection_error=registration.median_reprojection_error,
        valid_fraction=metrics["valid_fraction"],
        changed_fraction=metrics["changed_fraction"],
        stable_fraction=metrics["stable_fraction"],
        tiles_changed_fraction=metrics["tiles_changed_fraction"],
        largest_change_share=metrics["largest_change_share"],
        branch=branch,
    )
