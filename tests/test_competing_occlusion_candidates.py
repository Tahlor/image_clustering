"""Regression tests for residual-versus-text occlusion candidate selection."""

import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_candidates import (
    _component_candidates,
    _select_competing_candidate,
)


def _candidate(support: np.ndarray) -> dict[str, object]:
    return {
        "bbox": (0, 0, 240, 180),
        "support": support,
        "rectangularity": 1.0,
        "boundary": 0.0,
        "full_page": 0,
    }


def test_dense_candidate_replaces_residual_with_large_capture_gain() -> None:
    valid = np.ones((18, 24), dtype=bool)
    mismatch = np.zeros_like(valid)
    mismatch[4:14, 2:22] = True
    residual_support = np.zeros_like(valid)
    residual_support[4:8, 2:22] = True
    dense_support = np.zeros_like(valid)
    dense_support[4:14, 2:22] = True

    selected = _select_competing_candidate(
        _candidate(residual_support),
        _candidate(dense_support),
        mismatch,
        valid,
    )

    assert selected is not None
    assert np.array_equal(selected["support"], dense_support)


def test_dense_candidate_wins_at_exact_capture_boundary() -> None:
    valid = np.ones((18, 24), dtype=bool)
    mismatch = np.zeros_like(valid)
    mismatch[4:14, 2:22] = True
    residual_support = np.zeros_like(valid)
    residual_support[4:12, 2:22] = True
    dense_support = np.zeros_like(valid)
    dense_support[4:14, 2:22] = True

    selected = _select_competing_candidate(
        _candidate(residual_support),
        _candidate(dense_support),
        mismatch,
        valid,
    )

    assert selected is not None
    assert np.array_equal(selected["support"], dense_support)


def test_residual_candidate_wins_when_capture_gain_is_small() -> None:
    valid = np.ones((18, 24), dtype=bool)
    mismatch = np.zeros_like(valid)
    mismatch[4:14, 2:22] = True
    residual_support = np.zeros_like(valid)
    residual_support[4:13, 2:22] = True
    dense_support = np.zeros_like(valid)
    dense_support[4:14, 2:22] = True

    selected = _select_competing_candidate(
        _candidate(residual_support),
        _candidate(dense_support),
        mismatch,
        valid,
    )

    assert selected is not None
    assert np.array_equal(selected["support"], residual_support)


def test_full_page_residual_cannot_be_displaced() -> None:
    valid = np.ones((18, 24), dtype=bool)
    mismatch = np.zeros_like(valid)
    mismatch[4:14, 2:22] = True
    dense_support = np.zeros_like(valid)
    dense_support[4:14, 2:22] = True

    selected = _select_competing_candidate(
        _candidate(valid.copy()),
        _candidate(dense_support),
        mismatch,
        valid,
    )

    assert selected is not None
    assert np.array_equal(selected["support"], valid)


def test_component_selector_uses_dense_block_over_sparse_residual() -> None:
    config = ClusterConfig()
    scores = np.zeros((18, 24), dtype=np.float32)
    changed = np.zeros((18, 24), dtype=bool)
    changed[4:8, 2:22] = True
    scores[changed] = 1.0
    valid = np.ones((18, 24), dtype=bool)
    mismatch = np.zeros_like(valid)
    mismatch[4:14, 2:22] = True
    reference = np.full((180, 240), 238, dtype=np.uint8)

    selected = _component_candidates(
        scores=scores,
        changed=changed,
        valid_tiles=valid,
        ink_tile_mismatch=mismatch,
        page_columns=range(24),
        page_bbox=(0, 0, 240, 180),
        shape=reference.shape,
        reference=reference,
        aligned=reference.copy(),
        config=config,
    )

    assert len(selected) == 1
    support = selected[0]["support"]
    assert isinstance(support, np.ndarray)
    expected = np.zeros_like(valid)
    expected[4:14, 2:22] = True
    assert np.array_equal(support, expected)


def test_component_selector_retains_strong_residual() -> None:
    config = ClusterConfig()
    scores = np.zeros((18, 24), dtype=np.float32)
    changed = np.zeros((18, 24), dtype=bool)
    changed[4:13, 2:22] = True
    scores[changed] = 1.0
    valid = np.ones((18, 24), dtype=bool)
    mismatch = np.zeros_like(valid)
    mismatch[4:14, 2:22] = True
    reference = np.full((180, 240), 238, dtype=np.uint8)

    selected = _component_candidates(
        scores=scores,
        changed=changed,
        valid_tiles=valid,
        ink_tile_mismatch=mismatch,
        page_columns=range(24),
        page_bbox=(0, 0, 240, 180),
        shape=reference.shape,
        reference=reference,
        aligned=reference.copy(),
        config=config,
    )

    assert len(selected) == 1
    support = selected[0]["support"]
    assert isinstance(support, np.ndarray)
    expected = np.zeros_like(valid)
    expected[4:13, 2:22] = True
    assert np.array_equal(support, expected)
