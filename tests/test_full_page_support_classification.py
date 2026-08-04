"""Tests for source-independent full-page candidate classification."""

import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_models import ContentGrid
from image_clustering.clustering.content_summary import (
    _candidate_page_support_fraction,
    build_content_metrics,
)


def _grid() -> ContentGrid:
    core = np.ones((180, 240), dtype=bool)
    mismatch = np.zeros_like(core)
    ink_union = np.ones_like(core)
    valid_tiles = np.ones((18, 24), dtype=bool)
    scores = np.zeros((18, 24), dtype=np.float32)
    return ContentGrid(
        core=core,
        mismatch=mismatch,
        ink_union=ink_union,
        component_areas=[],
        unmatched_fraction=0.0,
        unmatched_union_fraction=0.0,
        largest_component_fraction=0.0,
        scores=scores,
        valid_tiles=valid_tiles,
        ink_tile_mismatch=np.zeros_like(valid_tiles),
        zscores=scores.copy(),
        changed=np.zeros_like(valid_tiles),
        residual_tiles_changed_fraction=0.0,
        ink_mismatch_tiles_fraction=0.0,
    )


def _candidate(support: np.ndarray, bbox: tuple[int, int, int, int]):
    return {
        "bbox": bbox,
        "support": support,
        "rectangularity": 1.0,
        "boundary": 0.0,
        "full_page": 0,
    }


def test_near_total_dense_support_is_classified_as_full_page() -> None:
    grid = _grid()
    support = np.zeros_like(grid.valid_tiles)
    support[2:16, :] = True
    reference = np.full((180, 240), 238, dtype=np.uint8)

    metrics = build_content_metrics(
        reference=reference,
        aligned=reference.copy(),
        grid=grid,
        selected=[_candidate(support, (0, 20, 240, 160))],
        page_regions=[(range(24), (0, 0, 240, 180))],
        gutter=None,
        config=ClusterConfig(),
    )

    assert metrics.full_page_occlusion_count == 1


def test_partial_support_remains_localized() -> None:
    grid = _grid()
    support = np.zeros_like(grid.valid_tiles)
    support[3:15, :] = True
    reference = np.full((180, 240), 238, dtype=np.uint8)

    metrics = build_content_metrics(
        reference=reference,
        aligned=reference.copy(),
        grid=grid,
        selected=[_candidate(support, (0, 30, 240, 150))],
        page_regions=[(range(24), (0, 0, 240, 180))],
        gutter=None,
        config=ClusterConfig(),
    )

    assert metrics.full_page_occlusion_count == 0


def test_support_fraction_is_page_local_on_a_spread() -> None:
    valid = np.ones((18, 24), dtype=bool)
    support = np.zeros_like(valid)
    support[:, :10] = True

    fraction = _candidate_page_support_fraction(
        support=support,
        valid_tiles=valid,
        bbox=(0, 0, 100, 180),
        gutter=120,
        image_width=240,
    )

    assert fraction == 10 / 12


def test_explicit_full_page_flag_remains_authoritative() -> None:
    grid = _grid()
    support = np.zeros_like(grid.valid_tiles)
    support[7:11, 2:22] = True
    candidate = _candidate(support, (20, 70, 220, 110))
    candidate["full_page"] = 1
    reference = np.full((180, 240), 238, dtype=np.uint8)

    metrics = build_content_metrics(
        reference=reference,
        aligned=reference.copy(),
        grid=grid,
        selected=[candidate],
        page_regions=[(range(24), (0, 0, 240, 180))],
        gutter=None,
        config=ClusterConfig(),
    )

    assert metrics.full_page_occlusion_count == 1
