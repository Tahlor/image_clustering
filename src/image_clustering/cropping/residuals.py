from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .models import BBox
from .registration import normalize_pair


@dataclass
class ResidualGrid:
    scores: np.ndarray
    zscores: np.ndarray
    changed: np.ndarray
    valid: np.ndarray
    boxes: list[BBox]
    tile_rows: int
    tile_cols: int
    baseline: float
    scale: float


def compute_residual_grid(
    reference_gray: np.ndarray,
    aligned_gray: np.ndarray,
    valid_mask: np.ndarray,
    config: Config,
) -> ResidualGrid:
    """Compute robust tile-level appearance disagreement."""
    section = config.section("residual")
    normalized = normalize_pair(reference_gray, aligned_gray, valid_mask)
    reference = reference_gray.astype(np.float32) / 255.0
    moving = normalized.astype(np.float32) / 255.0
    absolute = cv2.GaussianBlur(np.abs(reference - moving), (5, 5), 0)
    ref_gradient = cv2.magnitude(
        cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3),
    )
    mov_gradient = cv2.magnitude(
        cv2.Sobel(moving, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(moving, cv2.CV_32F, 0, 1, ksize=3),
    )
    gradient = np.clip(
        cv2.GaussianBlur(np.abs(ref_gradient - mov_gradient), (5, 5), 0),
        0,
        1,
    )

    height, width = reference.shape
    tile_size = max(
        12,
        round(max(height, width) / int(section["tiles_long_dimension"])),
    )
    rows = int(np.ceil(height / tile_size))
    cols = int(np.ceil(width / tile_size))
    scores = np.full((rows, cols), np.nan, dtype=np.float32)
    valid_tiles = np.zeros((rows, cols), dtype=bool)
    boxes: list[BBox] = []
    tail_fraction = float(section["upper_tail_fraction"])

    for row in range(rows):
        y0 = row * tile_size
        y1 = min(height, y0 + tile_size)
        for col in range(cols):
            x0 = col * tile_size
            x1 = min(width, x0 + tile_size)
            boxes.append((x0, y0, x1, y1))
            valid = valid_mask[y0:y1, x0:x1] > 0
            if valid.mean() < 0.72:
                continue
            absolute_values = absolute[y0:y1, x0:x1][valid]
            gradient_values = gradient[y0:y1, x0:x1][valid]
            if absolute_values.size < 20:
                continue
            tail_count = max(1, round(tail_fraction * absolute_values.size))
            upper_tail = float(
                np.partition(absolute_values, -tail_count)[-tail_count:].mean()
            )
            scores[row, col] = (
                float(section["absolute_weight"]) * float(absolute_values.mean())
                + float(section["upper_tail_weight"]) * upper_tail
                + float(section["gradient_weight"]) * float(gradient_values.mean())
            )
            valid_tiles[row, col] = True

    values = scores[valid_tiles]
    if values.size == 0:
        return ResidualGrid(
            scores,
            np.zeros_like(scores),
            np.zeros_like(valid_tiles),
            valid_tiles,
            boxes,
            rows,
            cols,
            0.0,
            1.0,
        )
    ordered = np.sort(values)
    stable_count = max(4, round(float(section["stable_fraction"]) * len(ordered)))
    stable = ordered[:stable_count]
    baseline = float(np.median(stable))
    mad = float(np.median(np.abs(stable - baseline)))
    robust_scale = max(1.4826 * mad, 0.012)
    zscores = np.zeros_like(scores)
    zscores[valid_tiles] = (scores[valid_tiles] - baseline) / robust_scale
    changed = valid_tiles & (
        (zscores >= float(section["changed_z_threshold"]))
        & (scores >= float(section["changed_min_absolute"]))
    )
    radius = int(section["close_radius_tiles"])
    if radius > 0:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        changed = cv2.morphologyEx(
            changed.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        ).astype(bool)
    changed &= valid_tiles
    return ResidualGrid(
        scores,
        zscores,
        changed,
        valid_tiles,
        boxes,
        rows,
        cols,
        baseline,
        robust_scale,
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
    index = np.searchsorted(cumulative, quantile * cumulative[-1])
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def component_rectangles(
    grid: ResidualGrid,
    config: Config,
    region_bbox: BBox | None = None,
) -> list[tuple[BBox, np.ndarray]]:
    """Return coherent tile components with low-mass tails trimmed."""
    mask = grid.changed.copy()
    if region_bbox is not None:
        rx0, ry0, rx1, ry1 = region_bbox
        for row in range(grid.tile_rows):
            for col in range(grid.tile_cols):
                box = grid.boxes[row * grid.tile_cols + col]
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                if not (rx0 <= center_x < rx1 and ry0 <= center_y < ry1):
                    mask[row, col] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    minimum = int(config.get("residual.min_component_tiles", 2))
    trim_x = float(config.get("residual.support_mass_trim_fraction_x", 0.0))
    trim_y = float(config.get("residual.support_mass_trim_fraction_y", 0.12))
    rectangles: list[tuple[BBox, np.ndarray]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        support = labels == label
        coordinates = list(zip(*np.where(support)))
        selected_boxes = [
            grid.boxes[row * grid.tile_cols + col] for row, col in coordinates
        ]
        centers_x = np.asarray(
            [(box[0] + box[2]) / 2 for box in selected_boxes],
            dtype=np.float32,
        )
        centers_y = np.asarray(
            [(box[1] + box[3]) / 2 for box in selected_boxes],
            dtype=np.float32,
        )
        weights = np.asarray(
            [
                max(float(grid.scores[row, col] - grid.baseline), 1e-4)
                for row, col in coordinates
            ],
            dtype=np.float32,
        )
        low_x = _weighted_quantile(centers_x, weights, trim_x)
        high_x = _weighted_quantile(centers_x, weights, 1.0 - trim_x)
        low_y = _weighted_quantile(centers_y, weights, trim_y)
        high_y = _weighted_quantile(centers_y, weights, 1.0 - trim_y)
        trimmed = [
            box
            for box, center_x, center_y in zip(
                selected_boxes,
                centers_x,
                centers_y,
            )
            if low_x <= center_x <= high_x and low_y <= center_y <= high_y
        ]
        if not trimmed:
            trimmed = selected_boxes
        bbox = (
            min(box[0] for box in trimmed),
            min(box[1] for box in trimmed),
            max(box[2] for box in trimmed),
            max(box[3] for box in trimmed),
        )
        rectangles.append((bbox, support))
    rectangles.sort(key=lambda item: int(item[1].sum()), reverse=True)
    return rectangles
