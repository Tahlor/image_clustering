"""Shared document-content representations for pair scoring."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig


def _detect_gutter(image: np.ndarray, config: ClusterConfig) -> int | None:
    height, width = image.shape
    if width < config.gutter_min_aspect_ratio * height:
        return None
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=max(2.0, width / 450))
    gradient = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    darkness = 1.0 - blurred.astype(np.float32).mean(axis=0) / 255.0
    edge_profile = gradient.mean(axis=0)
    edge_profile /= max(float(np.percentile(edge_profile, 95)), 1e-6)
    profile = 0.55 * darkness + 0.45 * edge_profile
    low = round(config.gutter_search_min_fraction * width)
    high = round(config.gutter_search_max_fraction * width)
    search = profile[low:high]
    if search.size == 0:
        return None
    index = low + int(np.argmax(search))
    prominence = float(profile[index] - np.median(search))
    return index if prominence >= config.gutter_min_prominence else None


def _boundary_score(image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    height, width = image.shape
    box_width = x1 - x0
    box_height = y1 - y0
    if box_width < 20 or box_height < 20:
        return 0.0
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gradient_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    band = max(1, round(min(box_width, box_height) * 0.006))
    search_x = max(8, round(box_width * 0.10))
    search_y = max(8, round(box_height * 0.10))

    def vertical_energy(x: int) -> float:
        xa = max(0, x - band)
        xb = min(width, x + band + 1)
        return float(np.mean(gradient_x[max(0, y0) : min(height, y1), xa:xb]))

    def horizontal_energy(y: int) -> float:
        ya = max(0, y - band)
        yb = min(height, y + band + 1)
        return float(np.mean(gradient_y[ya:yb, max(0, x0) : min(width, x1)]))

    def prominence(target: float, nearby: list[float]) -> float:
        return target / max(float(np.median(nearby)), 1.0) if nearby else 0.0

    x_offsets = range(3, search_x + 1, max(3, band * 2))
    y_offsets = range(3, search_y + 1, max(3, band * 2))
    ratios = np.asarray(
        [
            prominence(
                vertical_energy(x0),
                [
                    vertical_energy(x0 + offset)
                    for offset in x_offsets
                    if x0 + offset < x1
                ],
            ),
            prominence(
                vertical_energy(x1 - 1),
                [
                    vertical_energy(x1 - 1 - offset)
                    for offset in x_offsets
                    if x1 - 1 - offset > x0
                ],
            ),
            prominence(
                horizontal_energy(y0),
                [
                    horizontal_energy(y0 + offset)
                    for offset in y_offsets
                    if y0 + offset < y1
                ],
            ),
            prominence(
                horizontal_energy(y1 - 1),
                [
                    horizontal_energy(y1 - 1 - offset)
                    for offset in y_offsets
                    if y1 - 1 - offset > y0
                ],
            ),
        ],
        dtype=np.float32,
    )
    return float(np.sort(ratios)[1])


def _tile_bounds(
    row: int,
    column: int,
    shape: tuple[int, int],
    rows: int,
    columns: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    return (
        round(column * width / columns),
        round(row * height / rows),
        round((column + 1) * width / columns),
        round((row + 1) * height / rows),
    )


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 0:
        return float(np.quantile(values, quantile))
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1]))
    return float(ordered_values[min(index, len(ordered_values) - 1)])
