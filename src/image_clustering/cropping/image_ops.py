from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np

from .models import BBox


def gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def robust_normalize(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    low, high = np.percentile(values, [2.0, 98.0])
    if high <= low + 1.0:
        return np.zeros_like(image)
    normalized = np.clip((values - low) * (255.0 / (high - low)), 0, 255)
    return normalized.astype(np.uint8)


def fingerprint(image_gray: np.ndarray, size: int) -> np.ndarray:
    small = cv2.resize(image_gray, (size, size), interpolation=cv2.INTER_AREA)
    small = robust_normalize(small).astype(np.float32) / 255.0
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    vector = np.concatenate([small.ravel(), magnitude.ravel()])
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector /= norm
    return vector


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size:
        return -1.0
    return float(np.dot(a, b))


def affine_to_homography(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(np.float64)
    result = np.eye(3, dtype=np.float64)
    result[:2, :] = matrix
    return result


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(affine_to_homography(matrix))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    matrix_h = affine_to_homography(matrix)
    points_h = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (matrix_h @ points_h.T).T
    transformed[:, :2] /= np.maximum(transformed[:, 2:3], 1e-12)
    return transformed[:, :2]


def transform_bbox(bbox: BBox, matrix: np.ndarray, shape: tuple[int, int]) -> BBox:
    x0, y0, x1, y1 = bbox
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    transformed = transform_points(corners, matrix)
    height, width = shape
    return (
        max(0, int(math.floor(transformed[:, 0].min()))),
        max(0, int(math.floor(transformed[:, 1].min()))),
        min(width, int(math.ceil(transformed[:, 0].max()))),
        min(height, int(math.ceil(transformed[:, 1].max()))),
    )


def bbox_area(bbox: BBox) -> int:
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_intersection(a: BBox, b: BBox) -> BBox:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def bbox_iou(a: BBox, b: BBox) -> float:
    intersection = bbox_area(bbox_intersection(a, b))
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / max(union, 1)


def union_bbox(boxes: Iterable[BBox]) -> BBox:
    values = list(boxes)
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )
