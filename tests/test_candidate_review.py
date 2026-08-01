"""Tests for sequence-aware occlusion candidate ranking."""

from pathlib import Path

from image_clustering.clustering.candidate_review import rank_occlusion_candidates
from image_clustering.clustering.models import (
    ClusteringResult,
    ImageCluster,
    ImageItem,
    PairComparison,
)


def _pair(
    first: str,
    second: str,
    *,
    accepted: bool,
    score: float,
    contradiction: bool = False,
) -> PairComparison:
    return PairComparison(
        first_image_id=first,
        second_image_id=second,
        sequence_id="sequence",
        index_gap=1,
        same_document=accepted,
        confidence=0.8 if accepted else 0.49,
        reason="test",
        same_document_probability=0.8,
        occluded_given_same_probability=score / 0.8,
        same_occluded_probability=score,
        different_document_probability=0.2,
        occlusion_candidate_flag=True,
        automatic_link_eligible=accepted,
        hard_contradiction=contradiction,
    )


def _result() -> ClusteringResult:
    ids = ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg")
    images = tuple(
        ImageItem(image_id, Path(image_id), "sequence", index)
        for index, image_id in enumerate(ids)
    )
    comparisons = (
        _pair("a.jpg", "b.jpg", accepted=True, score=0.8),
        _pair("b.jpg", "c.jpg", accepted=True, score=0.8),
        _pair("a.jpg", "c.jpg", accepted=False, score=0.7),
        _pair("c.jpg", "d.jpg", accepted=False, score=0.9),
        _pair(
            "d.jpg",
            "e.jpg",
            accepted=False,
            score=0.95,
            contradiction=True,
        ),
    )
    clusters = (
        ImageCluster("cluster_00001", "sequence", ids[:3], "b.jpg"),
        ImageCluster("cluster_00002", "sequence", ("d.jpg",), "d.jpg"),
        ImageCluster("cluster_00003", "sequence", ("e.jpg",), "e.jpg"),
    )
    return ClusteringResult(
        config_fingerprint="test",
        images=images,
        clusters=clusters,
        comparisons=comparisons,
    )


def test_sequence_context_prioritizes_common_neighbor() -> None:
    candidates = rank_occlusion_candidates(_result())
    assert [candidate.review_tier for candidate in candidates] == [1, 3, 4]
    assert candidates[0].common_accepted_neighbors == ("b.jpg",)
    assert candidates[0].same_component


def test_hard_contradiction_never_becomes_link() -> None:
    candidate = rank_occlusion_candidates(_result())[-1]
    assert candidate.hard_contradiction
    assert candidate.review_tier == 4
    assert not candidate.automatic_link_eligible


def test_accepted_pairs_are_excluded_by_default() -> None:
    default = rank_occlusion_candidates(_result())
    with_accepted = rank_occlusion_candidates(_result(), include_accepted=True)
    assert len(default) == 3
    assert len(with_accepted) == 5
    assert with_accepted[0].review_tier == 0
