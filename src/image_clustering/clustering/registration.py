"""Feature matching and geometric registration."""

from __future__ import annotations

import math

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.models import ImageFeatures, Matrix3x3, Registration


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def _transform_is_plausible(
    matrix: np.ndarray,
    model: str,
    current_shape: tuple[int, int],
    previous_shape: tuple[int, int],
) -> bool:
    height, width = current_shape
    corners = np.float32(
        [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]]
    )
    transformed = (
        cv2.transform(corners, matrix)[0]
        if model == "affine"
        else cv2.perspectiveTransform(corners, matrix)[0]
    )
    if not np.isfinite(transformed).all():
        return False
    area_ratio = _polygon_area(transformed) / max(float(width * height), 1.0)
    previous_height, previous_width = previous_shape
    x_span = float(np.ptp(transformed[:, 0]))
    y_span = float(np.ptp(transformed[:, 1]))
    return (
        0.45 <= area_ratio <= 2.2
        and x_span >= 0.35 * previous_width
        and y_span >= 0.35 * previous_height
    )


def _evaluate_candidate(
    model: str,
    matrix: np.ndarray | None,
    inlier_mask: np.ndarray | None,
    current_points: np.ndarray,
    previous_points: np.ndarray,
    good_match_count: int,
    previous_feature_count: int,
    current_feature_count: int,
    current_shape: tuple[int, int],
    previous_shape: tuple[int, int],
    config: ClusterConfig,
) -> Registration | None:
    if matrix is None or inlier_mask is None:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(good_match_count, 1)
    feature_overlap = inlier_count / max(
        min(previous_feature_count, current_feature_count),
        1,
    )
    if inlier_count < config.min_inliers or inlier_ratio < config.min_inlier_ratio:
        return None
    if not _transform_is_plausible(
        matrix=matrix,
        model=model,
        current_shape=current_shape,
        previous_shape=previous_shape,
    ):
        return None

    inlier_previous = previous_points[inliers]
    previous_height, previous_width = previous_shape
    normalized = inlier_previous / np.float32([previous_width, previous_height])
    cells = {
        (
            min(3, max(0, int(x * 4))),
            min(3, max(0, int(y * 4))),
        )
        for x, y in normalized
    }
    x_span = float(np.ptp(inlier_previous[:, 0])) / max(previous_width, 1)
    y_span = float(np.ptp(inlier_previous[:, 1])) / max(previous_height, 1)
    if (
        len(cells) < config.min_grid_cells
        or x_span < config.min_x_span
        or y_span < config.min_y_span
    ):
        return None

    predicted = (
        cv2.transform(current_points[None], matrix)[0]
        if model == "affine"
        else cv2.perspectiveTransform(current_points[None], matrix)[0]
    )
    errors = np.linalg.norm(predicted - previous_points, axis=1)
    median_error = float(np.median(errors[inliers]))
    diagonal = math.hypot(*previous_shape)
    if median_error / max(diagonal, 1.0) > 0.015:
        return None
    return Registration(
        accepted=True,
        model=model,
        matrix=matrix.astype(np.float64),
        good_match_count=good_match_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        feature_overlap=feature_overlap,
        median_reprojection_error=median_error,
        occupied_grid_cells=len(cells),
        x_span=x_span,
        y_span=y_span,
    )


def register_pair(
    previous: ImageFeatures,
    current: ImageFeatures,
    config: ClusterConfig,
) -> Registration:
    """Register `current` into `previous` working-image coordinates."""
    if len(previous.descriptors) == 0 or len(current.descriptors) == 0:
        return Registration(accepted=False, reason="missing descriptors")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(current.descriptors, previous.descriptors, k=2)
    matches = [
        pair[0]
        for pair in raw_matches
        if len(pair) == 2
        and pair[0].distance < config.ratio_test * pair[1].distance
    ]
    if len(matches) < config.min_inliers:
        return Registration(
            accepted=False,
            good_match_count=len(matches),
            reason="insufficient descriptor matches",
        )
    current_points = np.float32(
        [current.keypoints_xy[match.queryIdx] for match in matches]
    )
    previous_points = np.float32(
        [previous.keypoints_xy[match.trainIdx] for match in matches]
    )

    affine, affine_mask = cv2.estimateAffine2D(
        current_points,
        previous_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reprojection_px,
        maxIters=2000,
        confidence=0.995,
        refineIters=10,
    )
    candidates: list[Registration] = []
    affine_result = _evaluate_candidate(
        model="affine",
        matrix=affine,
        inlier_mask=affine_mask,
        current_points=current_points,
        previous_points=previous_points,
        good_match_count=len(matches),
        previous_feature_count=len(previous.keypoints_xy),
        current_feature_count=len(current.keypoints_xy),
        current_shape=current.gray.shape,
        previous_shape=previous.gray.shape,
        config=config,
    )
    if affine_result is not None:
        candidates.append(affine_result)

    if affine_result is None or affine_result.inlier_ratio < 0.75:
        homography, homography_mask = cv2.findHomography(
            current_points,
            previous_points,
            cv2.RANSAC,
            config.ransac_reprojection_px,
            maxIters=2000,
            confidence=0.995,
        )
        homography_result = _evaluate_candidate(
            model="homography",
            matrix=homography,
            inlier_mask=homography_mask,
            current_points=current_points,
            previous_points=previous_points,
            good_match_count=len(matches),
            previous_feature_count=len(previous.keypoints_xy),
            current_feature_count=len(current.keypoints_xy),
            current_shape=current.gray.shape,
            previous_shape=previous.gray.shape,
            config=config,
        )
        if homography_result is not None:
            candidates.append(homography_result)

    if not candidates:
        return Registration(
            accepted=False,
            good_match_count=len(matches),
            reason="registration failed support, coverage, or plausibility checks",
        )
    return max(
        candidates,
        key=lambda candidate: (
            candidate.inlier_count,
            candidate.inlier_ratio,
            -candidate.median_reprojection_error,
        ),
    )


def warp_current(
    current_gray: np.ndarray,
    registration: Registration,
    previous_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the current image and validity mask into previous coordinates."""
    if registration.matrix is None or registration.model is None:
        raise ValueError("Cannot warp a rejected registration")
    height, width = previous_shape
    source_mask = np.full(current_gray.shape, 255, dtype=np.uint8)
    if registration.model == "affine":
        aligned = cv2.warpAffine(
            current_gray,
            registration.matrix[:2],
            (width, height),
        )
        valid = cv2.warpAffine(
            source_mask,
            registration.matrix[:2],
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
    else:
        aligned = cv2.warpPerspective(
            current_gray,
            registration.matrix,
            (width, height),
        )
        valid = cv2.warpPerspective(
            source_mask,
            registration.matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
    return aligned, valid


def source_pixel_transform(
    registration: Registration,
    previous_scale: float,
    current_scale: float,
) -> Matrix3x3:
    """Convert a working-image registration into source-pixel coordinates.

    The returned transform maps points in the current source image into the
    previous source image.
    """
    if registration.matrix is None or registration.model is None:
        raise ValueError("Cannot convert a rejected registration")
    if registration.model == "affine":
        working_matrix = np.vstack(
            [registration.matrix[:2], np.array([0.0, 0.0, 1.0])]
        )
    else:
        working_matrix = registration.matrix.copy()
    current_to_working = np.diag([current_scale, current_scale, 1.0])
    working_to_previous = np.diag(
        [1.0 / previous_scale, 1.0 / previous_scale, 1.0]
    )
    source_matrix = working_to_previous @ working_matrix @ current_to_working
    source_matrix /= source_matrix[2, 2]
    return tuple(tuple(float(value) for value in row) for row in source_matrix)
