"""Physical-occlusion candidate detection on coarse residual grids."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_geometry import _boundary_score, _tile_bounds


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
    """Return the strongest contiguous residual region for one detected page."""
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
    # Never promote a page merely because ink differs across it: two filled
    # copies of the same form have exactly that signature. A full-page shortcut
    # requires broad *material* change as well as broad residual support.
    if (
        page_changed_fraction >= config.occlusion_full_page_min_changed_fraction
        and page_material_fraction >= config.occlusion_full_page_min_material_fraction
    ) or (
        page_material_fraction
        >= config.occlusion_full_page_strong_material_fraction
        and page_changed_fraction
        >= config.occlusion_full_page_strong_material_min_changed_fraction
    ):
        candidates.append(
            {
                "bbox": page_bbox,
                "support": page_valid.copy(),
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
    page_width = px1 - px0
    page_height = py1 - py0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.occlusion_min_component_tiles:
            continue
        support = (labels == label) & page_valid
        coordinates = list(zip(*np.where(support), strict=True))
        boxes = [
            _tile_bounds(row, column, shape, rows, columns)
            for row, column in coordinates
        ]
        # Use the full connected component for geometry. Quantile trimming
        # silently removed the ends of genuine skewed/warped overlays and made
        # large occlusions look too small, while the final scorer already has
        # strict support-density and exterior-agreement gates.
        raw_bbox = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
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
        candidates.append(
            {
                "bbox": bbox,
                # Keep the connected support. The old implementation replaced
                # it with every tile in the padded bbox, which let sparse text
                # changes swallow clean form structure and masquerade as a
                # rectangular sheet.
                "support": support,
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
