"""Tests for conservative graph construction."""

from image_clustering.clustering.graph import build_clusters
from image_clustering.clustering.models import PairComparison


def make_comparison(
    first: str,
    second: str,
    accepted: bool,
) -> PairComparison:
    """Create a minimal pair comparison for graph tests."""
    return PairComparison(
        first_image_id=first,
        second_image_id=second,
        sequence_id=".",
        index_gap=1,
        same_document=accepted,
        confidence=0.9 if accepted else 0.1,
        reason="test",
    )


def test_rejected_similar_template_does_not_merge() -> None:
    image_ids = ["a.jpg", "b.jpg", "c.jpg"]
    comparisons = [
        make_comparison("a.jpg", "b.jpg", True),
        make_comparison("b.jpg", "c.jpg", False),
    ]
    clusters = build_clusters(
        sequence_id=".",
        image_ids=image_ids,
        comparisons=comparisons,
    )
    assert [cluster.image_ids for cluster in clusters] == [
        ("a.jpg", "b.jpg"),
        ("c.jpg",),
    ]


def test_complete_component_is_not_split_for_downstream_batching() -> None:
    image_ids = [f"{index}.jpg" for index in range(6)]
    comparisons = [
        make_comparison(image_ids[index], image_ids[index + 1], True)
        for index in range(len(image_ids) - 1)
    ]
    clusters = build_clusters(
        sequence_id=".",
        image_ids=image_ids,
        comparisons=comparisons,
    )
    assert [cluster.image_ids for cluster in clusters] == [tuple(image_ids)]


def test_page_wide_negative_blocks_transitive_bridge() -> None:
    image_ids = ["a.jpg", "b.jpg", "c.jpg"]
    first = make_comparison("a.jpg", "b.jpg", True)
    second = make_comparison("b.jpg", "c.jpg", True)
    contradiction = PairComparison(
        first_image_id="a.jpg",
        second_image_id="c.jpg",
        sequence_id=".",
        index_gap=2,
        same_document=False,
        confidence=0.1,
        reason="page-wide disagreement",
        registration_model="affine",
        valid_fraction=0.95,
        feature_overlap=0.10,
        changed_fraction=0.80,
    )
    clusters = build_clusters(
        sequence_id=".",
        image_ids=image_ids,
        comparisons=[first, second, contradiction],
    )
    assert [cluster.image_ids for cluster in clusters] == [
        ("a.jpg", "b.jpg"),
        ("c.jpg",),
    ]


def test_registration_failure_does_not_block_occlusion_bridge() -> None:
    image_ids = ["a.jpg", "b.jpg", "c.jpg"]
    first = make_comparison("a.jpg", "b.jpg", True)
    second = make_comparison("b.jpg", "c.jpg", True)
    no_direct_overlap = make_comparison("a.jpg", "c.jpg", False)
    clusters = build_clusters(
        sequence_id=".",
        image_ids=image_ids,
        comparisons=[first, second, no_direct_overlap],
    )
    assert [cluster.image_ids for cluster in clusters] == [tuple(image_ids)]
