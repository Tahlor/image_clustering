"""Regression tests for persisted localized physical-occlusion evidence."""

from pathlib import Path

import numpy as np

from image_clustering.clustering.candidate_review import _content_metrics
from image_clustering.clustering.candidate_scoring import PairProbabilities
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import (
    ImageFeatures,
    ImageItem,
    PairComparison,
    Registration,
)


def _content() -> ContentMetrics:
    return ContentMetrics(
        unmatched_ink_fraction=0.02,
        unmatched_ink_union_fraction=0.18,
        ink_mismatch_tiles_fraction=0.24,
        coherent_ink_component_count=4,
        largest_ink_component_fraction=0.03,
        residual_tiles_changed_fraction=0.38,
        occlusion_candidate_count=1,
        occlusion_area_fraction=0.42,
        occlusion_residual_capture=0.79,
        occlusion_rectangularity=0.71,
        occlusion_boundary_score=1.2,
        occlusion_material_fraction=0.66,
        occlusion_material_median=0.043,
        outside_unmatched_ink_fraction=0.003,
        outside_unmatched_ink_union_fraction=0.018,
        outside_ink_mismatch_tiles_fraction=0.06,
        full_page_occlusion_count=0,
        shallow_occlusion_count=0,
        page_count=1,
        inside_unmatched_ink_union_fraction=0.69,
        occlusion_ink_mismatch_capture=0.84,
        occlusion_localization_contrast=0.57,
    )


def _features(name: str, index: int) -> ImageFeatures:
    return ImageFeatures(
        image=ImageItem(
            image_id=name,
            path=Path(name),
            sequence_id="sequence",
            sequence_index=index,
        ),
        gray=np.full((32, 24), 240, dtype=np.uint8),
        scale=1.0,
        keypoints_xy=np.empty((0, 2), dtype=np.float32),
        descriptors=np.empty((0, 128), dtype=np.float32),
    )


def test_pair_comparison_round_trip_preserves_localization_diagnostics() -> None:
    comparison = PairComparison(
        first_image_id="a.jpg",
        second_image_id="b.jpg",
        sequence_id="sequence",
        index_gap=1,
        same_document=False,
        confidence=0.49,
        reason="review",
        inside_unmatched_ink_union_fraction=0.69,
        occlusion_ink_mismatch_capture=0.84,
        occlusion_localization_contrast=0.57,
        raw_occluded_given_same_probability=0.93,
        occluded_given_same_probability=0.61,
        occlusion_evidence=0.66,
    )

    restored = PairComparison.from_dict(comparison.to_dict())

    assert restored == comparison


def test_review_reconstruction_keeps_localization_diagnostics() -> None:
    comparison = PairComparison(
        first_image_id="a.jpg",
        second_image_id="b.jpg",
        sequence_id="sequence",
        index_gap=1,
        same_document=False,
        confidence=0.49,
        reason="review",
        inside_unmatched_ink_union_fraction=0.69,
        occlusion_ink_mismatch_capture=0.84,
        occlusion_localization_contrast=0.57,
    )

    content = _content_metrics(comparison)

    assert content.inside_unmatched_ink_union_fraction == 0.69
    assert content.occlusion_ink_mismatch_capture == 0.84
    assert content.occlusion_localization_contrast == 0.57


def test_score_pair_passes_runtime_config_and_persists_evidence(
    monkeypatch,
) -> None:
    from image_clustering.clustering import scoring as scoring_module

    content = _content()
    config = ClusterConfig(
        occlusion_evidence_min_ink_mismatch_capture=0.22,
        occlusion_evidence_full_ink_mismatch_capture=0.88,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        scoring_module,
        "register_pair",
        lambda **kwargs: Registration(
            accepted=True,
            model="affine",
            matrix=np.eye(2, 3),
            feature_overlap=0.24,
            inlier_ratio=0.75,
            x_span=0.8,
            y_span=0.8,
        ),
    )
    monkeypatch.setattr(
        scoring_module,
        "warp_current",
        lambda **kwargs: (
            np.full((32, 24), 240, dtype=np.uint8),
            np.full((32, 24), 255, dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        scoring_module,
        "_change_metrics",
        lambda **kwargs: {
            "valid_fraction": 1.0,
            "changed_fraction": 0.42,
            "stable_fraction": 0.58,
            "tiles_changed_fraction": 0.40,
            "largest_change_share": 0.70,
        },
    )
    monkeypatch.setattr(scoring_module, "analyze_content", lambda **kwargs: content)
    monkeypatch.setattr(
        scoring_module,
        "_decision",
        lambda **kwargs: (False, None, "review candidate"),
    )
    monkeypatch.setattr(
        scoring_module,
        "_hard_contradiction",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(scoring_module, "_confidence", lambda **kwargs: 0.49)
    monkeypatch.setattr(
        scoring_module,
        "source_pixel_transform",
        lambda **kwargs: None,
    )

    def fake_probabilities(**kwargs):
        captured["config"] = kwargs["config"]
        return PairProbabilities(
            same_document=0.90,
            occluded_given_same=0.63,
            same_clean=0.333,
            same_occluded=0.567,
            different_document=0.10,
            candidate_flag=True,
            automatic_link_eligible=False,
            raw_occluded_given_same=0.90,
            occlusion_evidence=0.70,
        )

    monkeypatch.setattr(scoring_module, "pair_probabilities", fake_probabilities)

    comparison = scoring_module.score_pair(
        previous=_features("a.jpg", 0),
        current=_features("b.jpg", 1),
        index_gap=1,
        config=config,
    )

    assert captured["config"] is config
    assert comparison.inside_unmatched_ink_union_fraction == 0.69
    assert comparison.occlusion_ink_mismatch_capture == 0.84
    assert comparison.occlusion_localization_contrast == 0.57
    assert comparison.raw_occluded_given_same_probability == 0.90
    assert comparison.occlusion_evidence == 0.70
