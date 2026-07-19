"""Physical-occlusion candidate detection on coarse residual grids."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_geometry import (
    _boundary_score,
    _tile_bounds,
    _weighted_quantile,
)


def _component_candidates(
    scores: np.ndarray,
    changed: np.ndarray,
    valid_tiles: np.ndarray,
    ink_tile_mismatch: np.ndarray,
    page_columns: range,
    page_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    reference: np.ndarray,
    aligned: np.ndarray,
    config: ClusterConfig,
) -> list[dict[str, float | int | tuple[int, int, int, int] | np.ndarray]]:
    page_mask = np.zeros_like(changed)
    page_mask[:, page_columns] = True
    page_valid = valid_tiles & page_mask
    if not page_valid.any():
        return []
    page_changed = changed & page_mask
    page_changed_fraction = float(page_changed.sum() / max(page_valid.sum(), 1))
    candidates: list[
        dict[str, float | int | tuple[int, int, int, int] | np.ndarray]
    ] = []
    page_ink_mismatch_fraction = float(
        (ink_tile_mismatch & page_valid).sum() / max(page_valid.sum(), 1)
    )
    px0, py0, px1, py1 = page_bbox
    material_sigma = max(3.0, min(reference.shape) / 120.0)
    page_material = np.abs(
        cv2.GaussianBlur(
            reference.astype(np.float32) / 255.0,
            (0, 0),
            material_sigma,
        )
        - cv2.GaussianBlur(
            aligned.astype(np.float32) / 255.0,
            (0, 0),
            material_sigma,
        )
    )[py0:py1, px0:px1]
    page_material_fraction = float(np.mean(page_material >= 0.04))
    if (
        page_changed_fraction >= config.occlusion_full_page_tile_fraction
        or (
            page_ink_mismatch_fraction >= config.occlusion_full_page_ink_tile_fraction
            and page_changed_fraction >= 0.12
        )
        or (
            page_material_fraction >= config.occlusion_full_page_material_fraction
            and page_changed_fraction <= config.occlusion_full_page_low_changed_fraction
            and page_ink_mismatch_fraction
            >= config.occlusion_full_page_min_ink_mismatch_fraction
        )
    ):
        support = page_valid.copy()
        candidates.append(
            {
                "bbox": page_bbox,
                "support": support,
                "rectangularity": 1.0,
                "boundary": 0.0,
                "full_page": 1,
            }
        )
        return candidates

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        page_changed.astype(np.uint8),
        connectivity=8,
    )
    rows, columns = changed.shape
    px0, py0, px1, py1 = page_bbox
    page_width = px1 - px0
    page_height = py1 - py0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.occlusion_min_component_tiles:
            continue
        support = labels == label
        coordinates = list(zip(*np.where(support), strict=True))
        boxes = [
            _tile_bounds(row, column, shape, rows, columns)
            for row, column in coordinates
        ]
        centers_x = np.asarray(
            [(box[0] + box[2]) / 2 for box in boxes],
            dtype=np.float32,
        )
        centers_y = np.asarray(
            [(box[1] + box[3]) / 2 for box in boxes],
            dtype=np.float32,
        )
        score_baseline = float(np.nanmedian(scores[valid_tiles]))
        weights = np.asarray(
            [
                max(float(scores[row, column] - score_baseline), 1e-4)
                for row, column in coordinates
            ],
            dtype=np.float32,
        )
        low_x = _weighted_quantile(centers_x, weights, 0.08)
        high_x = _weighted_quantile(centers_x, weights, 0.92)
        low_y = _weighted_quantile(centers_y, weights, 0.20)
        high_y = _weighted_quantile(centers_y, weights, 0.80)
        trimmed_boxes = [
            box
            for box, center_x, center_y in zip(boxes, centers_x, centers_y, strict=True)
            if low_x <= center_x <= high_x and low_y <= center_y <= high_y
        ]
        if not trimmed_boxes:
            trimmed_boxes = boxes
        raw_bbox = (
            min(box[0] for box in trimmed_boxes),
            min(box[1] for box in trimmed_boxes),
            max(box[2] for box in trimmed_boxes),
            max(box[3] for box in trimmed_boxes),
        )
        raw_width = raw_bbox[2] - raw_bbox[0]
        raw_height = raw_bbox[3] - raw_bbox[1]
        raw_area_fraction = raw_width * raw_height / max(page_width * page_height, 1)
        if raw_area_fraction < config.occlusion_min_page_area_fraction:
            continue
        padding_x = round(config.occlusion_padding_x_fraction * page_width)
        padding_y = round(config.occlusion_padding_y_fraction * page_height)
        bbox = (
            max(px0, raw_bbox[0] - padding_x),
            max(py0, raw_bbox[1] - padding_y),
            min(px1, raw_bbox[2] + padding_x),
            min(py1, raw_bbox[3] + padding_y),
        )
        tile_bbox_area = max(
            int(stats[label, cv2.CC_STAT_WIDTH])
            * int(stats[label, cv2.CC_STAT_HEIGHT]),
            1,
        )
        boundary = max(
            _boundary_score(reference, raw_bbox),
            _boundary_score(aligned, raw_bbox),
            _boundary_score(reference, bbox),
            _boundary_score(aligned, bbox),
        )
        bbox_support = np.zeros_like(support)
        for tile_row in range(rows):
            for tile_column in range(columns):
                tx0, ty0, tx1, ty1 = _tile_bounds(
                    tile_row,
                    tile_column,
                    shape,
                    rows,
                    columns,
                )
                center_x = (tx0 + tx1) / 2
                center_y = (ty0 + ty1) / 2
                if bbox[0] <= center_x < bbox[2] and bbox[1] <= center_y < bbox[3]:
                    bbox_support[tile_row, tile_column] = valid_tiles[
                        tile_row, tile_column
                    ]
        candidates.append(
            {
                "bbox": bbox,
                "support": bbox_support,
                "rectangularity": area / tile_bbox_area,
                "boundary": boundary,
                "full_page": 0,
            }
        )
    candidates.sort(
        key=lambda candidate: float(
            np.maximum(scores - np.nanmedian(scores[valid_tiles]), 0.0)[
                candidate["support"]
            ].sum()
        ),
        reverse=True,
    )
    return candidates[:1]
