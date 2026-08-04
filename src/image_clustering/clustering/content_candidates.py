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

_DENSE_INK_MIN_CAPTURE_GAIN = 0.20


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
    """Infer a sheet-shaped block from dense contiguous text erasure.

    A near-background sheet can erase print and handwriting while producing only a
    fragmented grayscale residual. This candidate requires a dense contiguous band
    in both row and column projections. The final scorer still requires block
    replacement, mismatch capture, registration support, and exterior agreement.
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


def _candidate_ink_capture(
    candidate: Candidate,
    ink_tile_mismatch: np.ndarray,
    page_valid: np.ndarray,
) -> float:
    """Return the page's text mismatch captured by one candidate support."""
    page_mismatch = ink_tile_mismatch & page_valid
    total_mismatch = int(page_mismatch.sum())
    if total_mismatch == 0:
        return 0.0
    support = candidate["support"]
    assert isinstance(support, np.ndarray)
    return float((page_mismatch & support & page_valid).sum() / total_mismatch)


def _select_competing_candidate(
    residual_candidate: Candidate | None,
    dense_ink_candidate: Candidate | None,
    ink_tile_mismatch: np.ndarray,
    page_valid: np.ndarray,
) -> Candidate | None:
    """Prefer dense text evidence only when it explains substantially more change.

    Residual structure remains the default because it carries direct material-change
    evidence. A dense text block replaces it only when it captures at least twenty
    percentage points more of the page's unmatched text. This recovers low-contrast
    overlays without allowing a slightly larger handwriting rectangle to displace a
    coherent physical residual.
    """
    if residual_candidate is None:
        return dense_ink_candidate
    if dense_ink_candidate is None:
        return residual_candidate
    residual_capture = _candidate_ink_capture(
        residual_candidate,
        ink_tile_mismatch,
        page_valid,
    )
    dense_capture = _candidate_ink_capture(
        dense_ink_candidate,
        ink_tile_mismatch,
        page_valid,
    )
    if dense_capture >= residual_capture + _DENSE_INK_MIN_CAPTURE_GAIN:
        return dense_ink_candidate
    return residual_candidate


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
    """Return the best localized physical-occlusion candidate for one page."""
    page_mask = np.zeros_like(changed)
    page_mask[:, page_columns] = True
    page_valid = valid_tiles & page_mask
    if not page_valid.any():
        return []
    page_changed = changed & page_mask
    page_changed_fraction = float(page_changed.sum() / max(page_valid.sum(), 1))
    residual_candidates: list[Candidate] = []
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
        residual_candidates.append(
            {
                "bbox": page_bbox,
                "support": page_valid.copy(),
                "rectangularity": 1.0,
                "boundary": 0.0,
                "full_page": 1,
            }
        )
    else:
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
            raw_bbox = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            raw_width = raw_bbox[2] - raw_bbox[0]
            raw_height = raw_bbox[3] - raw_bbox[1]
            raw_area_fraction = raw_width * raw_height / max(
                page_width * page_height,
                1,
            )
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
            residual_candidates.append(
                {
                    "bbox": bbox,
                    "support": support,
                    "rectangularity": area / tile_bbox_area,
                    "boundary": boundary,
                    "full_page": 0,
                    "residual_score": float(
                        np.maximum(
                            scores - np.nanmedian(scores[valid_tiles]),
                            0.0,
                        )[support].sum()
                    ),
                }
            )
        residual_candidates.sort(
            key=lambda candidate: float(candidate.get("residual_score", 0.0)),
            reverse=True,
        )

    residual_candidate = (
        residual_candidates[0] if residual_candidates else None
    )
    dense_candidate = _dense_ink_candidate(
        ink_tile_mismatch=ink_tile_mismatch,
        page_valid=page_valid,
        page_columns=page_columns,
        page_bbox=page_bbox,
        shape=shape,
        reference=reference,
        aligned=aligned,
        config=config,
    )
    selected = _select_competing_candidate(
        residual_candidate,
        dense_candidate,
        ink_tile_mismatch,
        page_valid,
    )
    return [selected] if selected is not None else []
