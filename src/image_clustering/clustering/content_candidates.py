"""Physical-occlusion candidate detection on coarse residual grids."""

from __future__ import annotations

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_geometry import _boundary_score, _tile_bounds

Candidate = dict[
    str,
    float | int | tuple[int, int, int, int] | np.ndarray,
]


def _runs_with_small_gaps(mask: np.ndarray, maximum_gap: int) -> list[tuple[int, int]]:
    """Return true runs after filling only very small internal gaps."""
    closed = mask.astype(bool).copy()
    true_indices = np.flatnonzero(closed)
    for first, second in zip(true_indices[:-1], true_indices[1:], strict=True):
        gap = int(second - first - 1)
        if 0 < gap <= maximum_gap:
            closed[first : second + 1] = True

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(closed):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(closed) - 1):
            end = index if value and index == len(closed) - 1 else index - 1
            runs.append((start, end))
            start = None
    return runs


def _dense_ink_candidate(
    ink_tile_mismatch: np.ndarray,
    page_valid: np.ndarray,
    page_columns: range,
    page_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    reference: np.ndarray,
    aligned: np.ndarray,
    config: ClusterConfig,
) -> Candidate | None:
    """Infer a sheet-shaped text-erasure block when grayscale residuals fragment.

    This is deliberately a fallback, not a second automatic acceptance path. It
    requires a dense contiguous band in both row and column projections. Only after
    that evidence is established is the interior rectangle filled as candidate
    support. The final scorer still requires block replacement, mismatch capture,
    registration support, and clean exterior agreement; distributed record text is
    rejected by those later gates.
    """
    columns = list(page_columns)
    page_ink = ink_tile_mismatch[:, columns] & page_valid[:, columns]
    page_valid_local = page_valid[:, columns]
    if not page_valid_local.any():
        return None

    row_denominator = np.maximum(page_valid_local.sum(axis=1), 1)
    row_density = page_ink.sum(axis=1) / row_denominator
    row_mask = row_density >= config.occlusion_ink_block_row_density

    rows, total_columns = ink_tile_mismatch.shape
    page_valid_count = int(page_valid.sum())
    candidates: list[tuple[float, Candidate]] = []
    for row_start, row_end in _runs_with_small_gaps(
        row_mask,
        config.occlusion_ink_block_max_gap,
    ):
        row_count = row_end - row_start + 1
        if row_count < config.occlusion_ink_block_min_rows:
            continue

        row_slice = slice(row_start, row_end + 1)
        column_denominator = np.maximum(
            page_valid_local[row_slice].sum(axis=0),
            1,
        )
        column_density = page_ink[row_slice].sum(axis=0) / column_denominator
        column_mask = (
            column_density >= config.occlusion_ink_block_column_density
        )
        for local_start, local_end in _runs_with_small_gaps(
            column_mask,
            config.occlusion_ink_block_max_gap,
        ):
            column_count = local_end - local_start + 1
            if column_count < (
                config.occlusion_ink_block_min_width_fraction * len(columns)
            ):
                continue
            column_start = columns[local_start]
            column_end = columns[local_end]
            support = np.zeros_like(page_valid)
            support[
                row_start : row_end + 1,
                column_start : column_end + 1,
            ] = True
            support &= page_valid
            support_count = int(support.sum())
            if support_count == 0:
                continue
            occupancy = float(
                (ink_tile_mismatch & support).sum() / support_count
            )
            if occupancy < config.occlusion_ink_block_min_occupancy:
                continue
            area_fraction = support_count / max(page_valid_count, 1)
            if area_fraction < config.occlusion_min_page_area_fraction:
                continue

            coordinates = list(zip(*np.where(support), strict=True))
            boxes = [
                _tile_bounds(row, column, shape, rows, total_columns)
                for row, column in coordinates
            ]
            bbox = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            px0, py0, px1, py1 = page_bbox
            bbox = (
                max(px0, bbox[0]),
                max(py0, bbox[1]),
                min(px1, bbox[2]),
                min(py1, bbox[3]),
            )
            boundary = max(
                _boundary_score(reference, bbox),
                _boundary_score(aligned, bbox),
            )
            candidate: Candidate = {
                "bbox": bbox,
                "support": support,
                "rectangularity": occupancy,
                "boundary": boundary,
                "full_page": 0,
            }
            candidates.append((occupancy * area_fraction, candidate))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


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
) -> list[Candidate]:
    """Return the strongest contiguous residual region for one detected page."""
    page_mask = np.zeros_like(changed)
    page_mask[:, page_columns] = True
    page_valid = valid_tiles & page_mask
    if not page_valid.any():
        return []
    page_changed = changed & page_mask
    page_changed_fraction = float(page_changed.sum() / max(page_valid.sum(), 1))
    candidates: list[Candidate] = []
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
        return [
            {
                "bbox": page_bbox,
                "support": page_valid.copy(),
                "rectangularity": 1.0,
                "boundary": 0.0,
                "full_page": 1,
            }
        ]

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
    if candidates:
        return candidates[:1]

    ink_candidate = _dense_ink_candidate(
        ink_tile_mismatch=ink_tile_mismatch,
        page_valid=page_valid,
        page_columns=page_columns,
        page_bbox=page_bbox,
        shape=shape,
        reference=reference,
        aligned=aligned,
        config=config,
    )
    return [ink_candidate] if ink_candidate is not None else []
