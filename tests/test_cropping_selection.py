from __future__ import annotations

import numpy as np

from image_clustering.cropping.config import Config
from image_clustering.cropping.selection import (
    choose_canonical_pages,
    choose_unique_region_states,
)


def config() -> Config:
    return Config(
        {
            "selection": {
                "complete_max_occlusion_fraction": 0.025,
                "partial_max_occlusion_fraction": 0.55,
                "min_page_frontness": 0.58,
                "canonical_frontness_weight": 0.20,
                "canonical_detail_weight": 0.80,
                "canonical_visibility_ratio": 0.90,
                "shallow_band_max_height_fraction": 0.27,
                "shallow_band_min_width_fraction": 0.52,
                "shallow_visibility_weight": 0.35,
                "submit_partial_best_available": True,
                "duplicate_correlation_threshold": 0.985,
                "min_content_score": 0.055,
                "min_region_frontness": 0.72,
                "page_state_min_page_coverage": 0.68,
                "blank_relative_detail_ratio": 0.67,
                "blank_relative_content_ratio": 0.65,
                "blank_relative_frontness_ratio": 0.72,
                "blank_boundary_override": 1.50,
                "min_unique_region_page_coverage": 0.25,
                "min_unique_state_change_fraction": 0.05,
                "min_boundary_score": 1.35,
            }
        }
    )


def test_reverse_only_page_is_not_submitted(monkeypatch) -> None:
    monkeypatch.setattr(
        "image_clustering.cropping.selection.page_frontness_score",
        lambda image, bbox: 0.40,
    )
    monkeypatch.setattr(
        "image_clustering.cropping.selection.absolute_detail_score",
        lambda image, bbox: 0.50,
    )
    monkeypatch.setattr(
        "image_clustering.cropping.selection.content_score",
        lambda image, bbox: 0.30,
    )
    images = {0: np.zeros((100, 100), np.uint8), 1: np.ones((100, 100), np.uint8)}
    submissions, canonical = choose_canonical_pages(
        group_id="g",
        page_boxes_anchor=[("single", (0, 0, 100, 100))],
        candidate_boxes_by_side={},
        transforms_to_anchor={0: np.eye(3), 1: np.eye(3)},
        image_shapes={0: (100, 100), 1: (100, 100)},
        images_gray=images,
        config=config(),
    )
    assert submissions == []
    assert set(canonical) == {"single"}
    assert canonical["single"] in {0, 1}


def test_wide_persistent_strip_marks_best_page_partial(monkeypatch) -> None:
    monkeypatch.setattr(
        "image_clustering.cropping.selection.page_frontness_score",
        lambda image, bbox: 1.0,
    )
    monkeypatch.setattr(
        "image_clustering.cropping.selection.absolute_detail_score",
        lambda image, bbox: 0.80,
    )
    monkeypatch.setattr(
        "image_clustering.cropping.selection.content_score",
        lambda image, bbox: 0.40,
    )
    images = {0: np.zeros((100, 100), np.uint8), 1: np.ones((100, 100), np.uint8)}
    submissions, _ = choose_canonical_pages(
        group_id="g",
        page_boxes_anchor=[("single", (0, 0, 100, 100))],
        candidate_boxes_by_side={"single": [(5, 40, 95, 60)]},
        transforms_to_anchor={0: np.eye(3), 1: np.eye(3)},
        image_shapes={0: (100, 100), 1: (100, 100)},
        images_gray=images,
        config=config(),
    )
    assert len(submissions) == 1
    assert submissions[0].completeness == "partial_best_available"


def test_two_distinct_foreground_states_are_both_retained(monkeypatch) -> None:
    images = {
        0: np.full((100, 100), 40, np.uint8),
        1: np.full((100, 100), 90, np.uint8),
        2: np.full((100, 100), 150, np.uint8),
    }

    def detail(image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        return {40: 0.55, 90: 0.70, 150: 0.85}[int(image[0, 0])]

    def content(image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        return {40: 0.42, 90: 0.55, 150: 0.68}[int(image[0, 0])]

    monkeypatch.setattr(
        "image_clustering.cropping.selection.page_frontness_score",
        lambda image, bbox: 1.0,
    )
    monkeypatch.setattr(
        "image_clustering.cropping.selection.absolute_detail_score",
        detail,
    )
    monkeypatch.setattr(
        "image_clustering.cropping.selection.content_score",
        content,
    )
    monkeypatch.setattr(
        "image_clustering.cropping.selection.boundary_score",
        lambda image, bbox: 2.0,
    )
    submissions = choose_unique_region_states(
        group_id="g",
        candidates=[("right", (50, 20, 100, 80))],
        page_boxes_anchor=[("right", (50, 0, 100, 100))],
        canonical_image_by_side={"right": 2},
        transforms_to_anchor={0: np.eye(3), 1: np.eye(3), 2: np.eye(3)},
        image_shapes={0: (100, 100), 1: (100, 100), 2: (100, 100)},
        images_gray=images,
        config=config(),
    )
    assert [submission.image_index for submission in submissions] == [1, 0]
    assert all(submission.kind == "data_bearing_overlay" for submission in submissions)
