"""Page partitioning and page-local occlusion candidate selection."""

from __future__ import annotations

import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_candidates import _component_candidates
from image_clustering.clustering.content_geometry import _detect_gutter
from image_clustering.clustering.content_models import ContentGrid

Candidate = dict[str, float | int | tuple[int, int, int, int] | np.ndarray]
PageRegion = tuple[range, tuple[int, int, int, int]]


def select_page_candidates(
    reference: np.ndarray,
    aligned: np.ndarray,
    grid: ContentGrid,
    config: ClusterConfig,
) -> tuple[list[Candidate], list[PageRegion], int | None]:
    """Partition a spread and detect at most one physical occlusion per page."""
    _, columns = grid.scores.shape
    gutter = _detect_gutter(reference, config)
    if gutter is None:
        page_regions: list[PageRegion] = [
            (range(columns), (0, 0, reference.shape[1], reference.shape[0]))
        ]
    else:
        gutter_column = min(
            columns - 1,
            max(1, round(gutter * columns / reference.shape[1])),
        )
        page_regions = [
            (range(0, gutter_column), (0, 0, gutter, reference.shape[0])),
            (
                range(gutter_column, columns),
                (gutter, 0, reference.shape[1], reference.shape[0]),
            ),
        ]

    selected: list[Candidate] = []
    for page_columns, page_bbox in page_regions:
        selected.extend(
            _component_candidates(
                scores=grid.scores,
                changed=grid.changed,
                valid_tiles=grid.valid_tiles,
                ink_tile_mismatch=grid.ink_tile_mismatch,
                page_columns=page_columns,
                page_bbox=page_bbox,
                shape=reference.shape,
                reference=reference,
                aligned=aligned,
                config=config,
            )
        )
    return selected, page_regions, gutter
