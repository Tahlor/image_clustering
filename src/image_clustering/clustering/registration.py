"""Feature matching and geometric registration."""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.models import ImageFeatures, Matrix3x3, Registration

_AFFINE_MODELS = {"affine", "ecc_euclidean"}
_ECC_LOCK = threading.Lock()


def _is_affine_model(model: str) -> bool:
    return model in _AFFINE_MODELS


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
        if _is_affine_model(model)
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
        if _is_affine_model(model)
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
        alignment_score=inlier_ratio,
    )


def _center_on_canvas(
    image: np.ndarray,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = canvas_shape
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    fill = int(np.median(border))
    canvas = np.full((height, width), fill, dtype=np.uint8)
    offset_x = (width - image.shape[1]) // 2
    offset_y = (height - image.shape[0]) // 2
    canvas[
        offset_y : offset_y + image.shape[0],
        offset_x : offset_x + image.shape[1],
    ] = image
    return canvas, (offset_x, offset_y)


def _ecc_preprocess(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32) / 255.0
    sigma = max(5.0, min(image.shape) * 0.03)
    background = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma)
    normalized = cv2.GaussianBlur(values - background, (5, 5), 0)
    normalized -= float(normalized.mean())
    normalized /= max(float(normalized.std()), 1e-6)
    return normalized.astype(np.float32)


def _bounded_small_motion_seed(
    matrix: np.ndarray | None,
    *,
    current_shape: tuple[int, int],
    previous_shape: tuple[int, int],
    config: ClusterConfig,
) -> bool:
    """Return whether an affine seed is compatible with production capture motion."""
    if matrix is None or matrix.shape not in {(2, 3), (3, 3)}:
        return False
    affine = matrix[:2]
    if not np.isfinite(affine).all():
        return False
    scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
    angle = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
    translation_x = abs(float(affine[0, 2])) / max(previous_shape[1], 1)
    translation_y = abs(float(affine[1, 2])) / max(previous_shape[0], 1)
    return (
        0.85 <= scale <= 1.18
        and abs(angle) <= config.ecc_max_rotation_degrees
        and translation_x <= config.ecc_max_translation_fraction
        and translation_y <= config.ecc_max_translation_fraction
        and _transform_is_plausible(
            matrix=affine,
            model="affine",
            current_shape=current_shape,
            previous_shape=previous_shape,
        )
    )


def _coarse_phase_response(
    previous: ImageFeatures,
    current: ImageFeatures,
    config: ClusterConfig,
) -> float:
    """Return cheap coarse translation evidence before attempting full ECC."""
    canvas_shape = (
        max(previous.gray.shape[0], current.gray.shape[0]),
        max(previous.gray.shape[1], current.gray.shape[1]),
    )
    previous_canvas, _ = _center_on_canvas(previous.gray, canvas_shape)
    current_canvas, _ = _center_on_canvas(current.gray, canvas_shape)
    scale = min(
        1.0,
        config.ecc_coarse_dimension / max(canvas_shape),
    )
    if scale < 1.0:
        size = (
            max(32, round(canvas_shape[1] * scale)),
            max(32, round(canvas_shape[0] * scale)),
        )
        previous_canvas = cv2.resize(
            previous_canvas,
            size,
            interpolation=cv2.INTER_AREA,
        )
        current_canvas = cv2.resize(
            current_canvas,
            size,
            interpolation=cv2.INTER_AREA,
        )
    template = _ecc_preprocess(previous_canvas)
    input_image = _ecc_preprocess(current_canvas)
    window = cv2.createHanningWindow(
        (template.shape[1], template.shape[0]),
        cv2.CV_32F,
    )
    try:
        _, response = cv2.phaseCorrelate(template, input_image, window)
    except cv2.error:
        return 0.0
    return float(response) if math.isfinite(float(response)) else 0.0


def _small_motion_ecc_registration(
    previous: ImageFeatures,
    current: ImageFeatures,
    config: ClusterConfig,
    initial_current_to_previous: np.ndarray | None = None,
) -> Registration:
    """Try a constrained same-orientation fallback for heavily occluded captures."""
    if not config.ecc_fallback_enabled:
        return Registration(accepted=False, reason="ECC fallback disabled")

    previous_aspect = previous.gray.shape[1] / max(previous.gray.shape[0], 1)
    current_aspect = current.gray.shape[1] / max(current.gray.shape[0], 1)
    aspect_ratio = previous_aspect / max(current_aspect, 1e-6)
    if not 0.85 <= aspect_ratio <= 1.18:
        return Registration(
            accepted=False,
            fallback_used=True,
            reason="ECC fallback rejected incompatible image aspect ratios",
        )

    canvas_shape = (
        max(previous.gray.shape[0], current.gray.shape[0]),
        max(previous.gray.shape[1], current.gray.shape[1]),
    )
    previous_canvas, previous_offset = _center_on_canvas(
        previous.gray,
        canvas_shape,
    )
    current_canvas, current_offset = _center_on_canvas(
        current.gray,
        canvas_shape,
    )

    # Full-resolution ECC can spend minutes on repetitive or unrelated forms.
    # Estimate motion on a bounded canvas, then map it back for full-resolution
    # content scoring.
    scale = min(1.0, config.ecc_working_dimension / max(canvas_shape))
    if scale < 1.0:
        working_size = (
            max(64, round(canvas_shape[1] * scale)),
            max(64, round(canvas_shape[0] * scale)),
        )
        previous_working = cv2.resize(
            previous_canvas,
            working_size,
            interpolation=cv2.INTER_AREA,
        )
        current_working = cv2.resize(
            current_canvas,
            working_size,
            interpolation=cv2.INTER_AREA,
        )
    else:
        previous_working = previous_canvas
        current_working = current_canvas
    scale_x = previous_working.shape[1] / canvas_shape[1]
    scale_y = previous_working.shape[0] / canvas_shape[0]
    canvas_to_working = np.diag([scale_x, scale_y, 1.0])

    template = _ecc_preprocess(previous_working)
    input_image = _ecc_preprocess(current_working)

    current_to_canvas = np.array(
        [
            [1.0, 0.0, current_offset[0]],
            [0.0, 1.0, current_offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    previous_to_canvas = np.array(
        [
            [1.0, 0.0, previous_offset[0]],
            [0.0, 1.0, previous_offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    current_to_working = canvas_to_working @ current_to_canvas
    previous_to_working = canvas_to_working @ previous_to_canvas
    warp: np.ndarray
    if initial_current_to_previous is not None:
        initial = (
            np.vstack(
                [initial_current_to_previous[:2], np.array([0.0, 0.0, 1.0])]
            )
            if initial_current_to_previous.shape == (2, 3)
            else initial_current_to_previous.copy()
        )
        current_to_previous_working = (
            previous_to_working @ initial @ np.linalg.inv(current_to_working)
        )
        warp = cv2.invertAffineTransform(
            current_to_previous_working[:2].astype(np.float32)
        )
    else:
        window = cv2.createHanningWindow(
            (template.shape[1], template.shape[0]),
            cv2.CV_32F,
        )
        try:
            (shift_x, shift_y), phase_response = cv2.phaseCorrelate(
                template,
                input_image,
                window,
            )
            if phase_response < config.ecc_min_phase_correlation:
                shift_x = shift_y = 0.0
        except cv2.error:
            shift_x = shift_y = 0.0
        warp = np.array(
            [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]],
            dtype=np.float32,
        )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_max_iterations,
        config.ecc_epsilon,
    )
    try:
        # OpenCV ECC is not reliably efficient when several calls run at once.
        # Keep the ordinary pair pipeline parallel and serialize this rare step.
        with _ECC_LOCK:
            correlation, template_to_current = cv2.findTransformECC(
                template,
                input_image,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                config.ecc_gaussian_filter_size,
            )
    except cv2.error:
        return Registration(
            accepted=False,
            fallback_used=True,
            reason="small-motion ECC fallback did not converge",
        )

    current_to_previous_working = cv2.invertAffineTransform(
        template_to_current
    )
    working_matrix = np.vstack(
        [current_to_previous_working, np.array([0.0, 0.0, 1.0])]
    )
    source_matrix = (
        np.linalg.inv(previous_to_working)
        @ working_matrix
        @ current_to_working
    )
    affine = source_matrix[:2].astype(np.float64)

    angle = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
    translation_x = abs(float(affine[0, 2])) / max(previous.gray.shape[1], 1)
    translation_y = abs(float(affine[1, 2])) / max(previous.gray.shape[0], 1)
    if (
        correlation < config.ecc_min_correlation
        or abs(angle) > config.ecc_max_rotation_degrees
        or translation_x > config.ecc_max_translation_fraction
        or translation_y > config.ecc_max_translation_fraction
        or not _transform_is_plausible(
            matrix=affine,
            model="ecc_euclidean",
            current_shape=current.gray.shape,
            previous_shape=previous.gray.shape,
        )
    ):
        return Registration(
            accepted=False,
            alignment_score=float(correlation),
            fallback_used=True,
            reason="small-motion ECC fallback failed correlation or transform bounds",
        )

    return Registration(
        accepted=True,
        model="ecc_euclidean",
        matrix=affine,
        inlier_ratio=float(correlation),
        alignment_score=float(correlation),
        occupied_grid_cells=16,
        x_span=1.0,
        y_span=1.0,
        fallback_used=True,
        reason="small-motion ECC fallback",
    )


def _fallback_with_match_count(
    previous: ImageFeatures,
    current: ImageFeatures,
    config: ClusterConfig,
    good_match_count: int,
    initial_current_to_previous: np.ndarray | None = None,
) -> Registration:
    seed_is_plausible = _bounded_small_motion_seed(
        initial_current_to_previous,
        current_shape=current.gray.shape,
        previous_shape=previous.gray.shape,
        config=config,
    )
    phase_response = _coarse_phase_response(previous, current, config)
    if (
        not seed_is_plausible
        and good_match_count < config.ecc_min_descriptor_matches
        and phase_response < config.ecc_min_phase_correlation
    ):
        return Registration(
            accepted=False,
            good_match_count=good_match_count,
            alignment_score=phase_response,
            reason=(
                "ECC fallback skipped: weak descriptor, affine-seed, and coarse "
                "phase evidence"
            ),
        )
    fallback = _small_motion_ecc_registration(
        previous=previous,
        current=current,
        config=config,
        initial_current_to_previous=(
            initial_current_to_previous if seed_is_plausible else None
        ),
    )
    fallback.good_match_count = good_match_count
    if fallback.alignment_score == 0.0 and phase_response > 0.0:
        fallback.alignment_score = phase_response
    return fallback


def register_pair(
    previous: ImageFeatures,
    current: ImageFeatures,
    config: ClusterConfig,
) -> Registration:
    """Register `current` into `previous` working-image coordinates."""
    if len(previous.descriptors) == 0 or len(current.descriptors) == 0:
        fallback = _fallback_with_match_count(previous, current, config, 0)
        if not fallback.accepted and fallback.reason is None:
            fallback.reason = "missing descriptors and ECC fallback failed"
        return fallback
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(current.descriptors, previous.descriptors, k=2)
    matches = [
        pair[0]
        for pair in raw_matches
        if len(pair) == 2
        and pair[0].distance < config.ratio_test * pair[1].distance
    ]
    current_points = np.float32(
        [current.keypoints_xy[match.queryIdx] for match in matches]
    )
    previous_points = np.float32(
        [previous.keypoints_xy[match.trainIdx] for match in matches]
    )
    loose_affine = None
    if len(matches) >= 4:
        loose_affine, _ = cv2.estimateAffinePartial2D(
            current_points,
            previous_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reprojection_px * 2.0,
            maxIters=1000,
            confidence=0.99,
            refineIters=5,
        )
    if len(matches) < config.min_inliers:
        return _fallback_with_match_count(
            previous,
            current,
            config,
            len(matches),
            loose_affine,
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
        return _fallback_with_match_count(
            previous,
            current,
            config,
            len(matches),
            affine if affine is not None else loose_affine,
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
    if _is_affine_model(registration.model):
        aligned = cv2.warpAffine(
            current_gray,
            registration.matrix[:2],
            (width, height),
            borderValue=255,
        )
        valid = cv2.warpAffine(
            source_mask,
            registration.matrix[:2],
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
    else:
        aligned = cv2.warpPerspective(
            current_gray,
            registration.matrix,
            (width, height),
            borderValue=255,
        )
        valid = cv2.warpPerspective(
            source_mask,
            registration.matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
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
    if _is_affine_model(registration.model):
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
