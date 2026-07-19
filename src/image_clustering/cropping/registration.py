from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .image_ops import affine_to_homography
from .models import FeatureSet, RegistrationResult


def detect_features(image_gray: np.ndarray, config: Config) -> FeatureSet:
    """Extract local features for robust registration."""
    detector_name = str(config.get("features.detector", "sift")).lower()
    if detector_name == "sift":
        detector = cv2.SIFT_create(
            nfeatures=int(config.get("features.nfeatures", 5000)),
            contrastThreshold=float(config.get("features.contrast_threshold", 0.018)),
        )
    elif detector_name == "akaze":
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported detector: {detector_name}")
    keypoints, descriptors = detector.detectAndCompute(image_gray, None)
    return FeatureSet(keypoints=keypoints, descriptors=descriptors)


def _matches(reference: FeatureSet, moving: FeatureSet, config: Config) -> list[cv2.DMatch]:
    if reference.descriptors is None or moving.descriptors is None:
        return []
    norm = cv2.NORM_L2 if moving.descriptors.dtype == np.float32 else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    raw = matcher.knnMatch(moving.descriptors, reference.descriptors, k=2)
    ratio = float(config.get("features.ratio_test", 0.76))
    return [pair[0] for pair in raw if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


def _points(
    reference: FeatureSet,
    moving: FeatureSet,
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    moving_points = np.float32([moving.keypoints[item.queryIdx].pt for item in matches])
    reference_points = np.float32([reference.keypoints[item.trainIdx].pt for item in matches])
    return moving_points, reference_points


def warp_to_reference(
    moving: np.ndarray,
    matrix: np.ndarray,
    reference_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image and a validity mask into the reference frame."""
    height, width = reference_shape
    matrix_h = affine_to_homography(matrix)
    aligned = cv2.warpPerspective(
        moving,
        matrix_h,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    source_mask = np.full(moving.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(
        source_mask,
        matrix_h,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = cv2.erode((valid > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    return aligned, valid


def normalize_pair(
    reference_gray: np.ndarray,
    aligned_gray: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Apply robust linear exposure normalization over valid overlap."""
    valid = valid_mask > 0
    if int(valid.sum()) < 1000:
        return aligned_gray
    reference = reference_gray[valid].astype(np.float32)
    moving = aligned_gray[valid].astype(np.float32)
    ref_median = float(np.median(reference))
    mov_median = float(np.median(moving))
    ref_mad = float(np.median(np.abs(reference - ref_median)))
    mov_mad = float(np.median(np.abs(moving - mov_median)))
    scale = 1.0 if min(ref_mad, mov_mad) < 1.0 else float(np.clip(ref_mad / mov_mad, 0.72, 1.38))
    offset = ref_median - scale * mov_median
    return np.clip(aligned_gray.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)


def _stable_residual(
    reference_gray: np.ndarray,
    aligned_gray: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    normalized = normalize_pair(reference_gray, aligned_gray, valid_mask)
    difference = cv2.GaussianBlur(
        np.abs(reference_gray.astype(np.float32) - normalized.astype(np.float32)) / 255.0,
        (7, 7),
        0,
    )
    height, width = reference_gray.shape
    tile = max(16, round(max(height, width) / 24))
    scores: list[float] = []
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1 = min(height, y0 + tile)
            x1 = min(width, x0 + tile)
            valid = valid_mask[y0:y1, x0:x1] > 0
            if valid.mean() < 0.75:
                continue
            values = difference[y0:y1, x0:x1][valid]
            if values.size:
                scores.append(float(values.mean()))
    if len(scores) < 8:
        return 1.0
    ordered = np.sort(np.asarray(scores, dtype=np.float32))
    keep = max(4, round(0.62 * len(ordered)))
    return float(ordered[:keep].mean())


def _coverage_ok(
    reference_points: np.ndarray,
    inlier_mask: np.ndarray,
    shape: tuple[int, int],
    config: Config,
) -> bool:
    selected = reference_points[inlier_mask.ravel().astype(bool)]
    if len(selected) < int(config.get("features.min_inliers", 14)):
        return False
    height, width = shape
    grid_size = int(config.get("features.inlier_grid_size", 4))
    occupied = {
        (
            min(grid_size - 1, max(0, int(x / max(width, 1) * grid_size))),
            min(grid_size - 1, max(0, int(y / max(height, 1) * grid_size))),
        )
        for x, y in selected
    }
    spans = np.ptp(selected, axis=0) / np.asarray([max(width, 1), max(height, 1)])
    return (
        len(occupied) >= int(config.get("features.min_inlier_grid_cells", 5))
        and spans[0] >= 0.30
        and spans[1] >= 0.24
    )


def _evaluate(
    reference_gray: np.ndarray,
    moving_gray: np.ndarray,
    matrix: np.ndarray | None,
    model: str,
    inlier_mask: np.ndarray | None,
    matches: list[cv2.DMatch],
    moving_points: np.ndarray,
    reference_points: np.ndarray,
    fingerprint_correlation: float,
    config: Config,
) -> RegistrationResult:
    if matrix is None or inlier_mask is None:
        return RegistrationResult(False, reason=f"{model} estimation failed")
    inlier_count = int(inlier_mask.ravel().sum())
    inlier_ratio = inlier_count / max(len(matches), 1)
    if inlier_ratio < float(config.get("features.min_inlier_ratio", 0.18)):
        return RegistrationResult(False, reason=f"{model}: low inlier ratio")
    if not _coverage_ok(reference_points, inlier_mask, reference_gray.shape, config):
        return RegistrationResult(False, reason=f"{model}: inliers lack spatial coverage")
    matrix_h = affine_to_homography(matrix)
    aligned, valid = warp_to_reference(moving_gray, matrix_h, reference_gray.shape)
    overlap = float((valid > 0).mean())
    if overlap < float(config.get("registration.min_overlap_fraction", 0.70)):
        return RegistrationResult(False, reason=f"{model}: insufficient overlap")
    residual = _stable_residual(reference_gray, aligned, valid)
    accepted = residual <= float(config.get("registration.max_stable_tile_residual", 0.19))
    return RegistrationResult(
        accepted=accepted,
        model=model,
        matrix=matrix_h,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        good_match_count=len(matches),
        overlap_fraction=overlap,
        stable_residual=residual,
        fingerprint_correlation=fingerprint_correlation,
        reason=None if accepted else f"{model}: stable residual {residual:.3f} too high",
    )


def register_pair(
    reference_gray: np.ndarray,
    moving_gray: np.ndarray,
    reference_features: FeatureSet,
    moving_features: FeatureSet,
    fingerprint_correlation: float,
    config: Config,
) -> RegistrationResult:
    """Register moving to reference, preferring affine over homography."""
    matches = _matches(reference_features, moving_features, config)
    if len(matches) < int(config.get("features.min_good_matches", 18)):
        return RegistrationResult(
            False,
            good_match_count=len(matches),
            fingerprint_correlation=fingerprint_correlation,
            reason=f"Only {len(matches)} good matches",
        )
    moving_points, reference_points = _points(reference_features, moving_features, matches)
    threshold = float(config.get("features.ransac_reprojection_px", 5.0))
    affine, affine_mask = cv2.estimateAffine2D(
        moving_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=4000,
        confidence=0.997,
        refineIters=12,
    )
    affine_result = _evaluate(
        reference_gray,
        moving_gray,
        affine,
        "affine",
        affine_mask,
        matches,
        moving_points,
        reference_points,
        fingerprint_correlation,
        config,
    )
    homography, homography_mask = cv2.findHomography(
        moving_points,
        reference_points,
        cv2.RANSAC,
        threshold,
        maxIters=4000,
        confidence=0.997,
    )
    homography_result = _evaluate(
        reference_gray,
        moving_gray,
        homography,
        "homography",
        homography_mask,
        matches,
        moving_points,
        reference_points,
        fingerprint_correlation,
        config,
    )
    if affine_result.accepted:
        improvement = affine_result.stable_residual - homography_result.stable_residual
        required = float(config.get("registration.homography_required_improvement", 0.018))
        if homography_result.accepted and improvement >= required:
            return homography_result
        return affine_result
    if homography_result.accepted:
        return homography_result
    return min(
        (affine_result, homography_result),
        key=lambda result: result.stable_residual,
    )
