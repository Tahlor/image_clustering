"""Tests for the public downstream handoff models."""

from pathlib import Path

from image_clustering import (
    ClusteringResult,
    ImageCluster,
    ImageItem,
    PairComparison,
)


def make_result() -> ClusteringResult:
    """Create a small complete clustering result."""
    images = (
        ImageItem("folder/a.jpg", Path("/data/folder/a.jpg"), "folder", 0),
        ImageItem("folder/b.jpg", Path("/data/folder/b.jpg"), "folder", 1),
    )
    comparison = PairComparison(
        first_image_id="folder/a.jpg",
        second_image_id="folder/b.jpg",
        sequence_id="folder",
        index_gap=1,
        same_document=True,
        confidence=0.9,
        reason="test",
        registration_model="affine",
        transform=((1.0, 0.0, 2.0), (0.0, 1.0, 3.0), (0.0, 0.0, 1.0)),
    )
    cluster = ImageCluster(
        cluster_id="cluster_00001",
        sequence_id="folder",
        image_ids=("folder/a.jpg", "folder/b.jpg"),
        representative_image_id="folder/a.jpg",
    )
    return ClusteringResult(
        config_fingerprint="abc",
        input_root=Path("/data"),
        images=images,
        clusters=(cluster,),
        comparisons=(comparison,),
    )


def test_result_exposes_cluster_images_and_registrations() -> None:
    result = make_result()
    assert [image.image_id for image in result.images_for("cluster_00001")] == [
        "folder/a.jpg",
        "folder/b.jpg",
    ]
    comparisons = result.accepted_comparisons("cluster_00001")
    assert len(comparisons) == 1
    assert comparisons[0].transform == (
        (1.0, 0.0, 2.0),
        (0.0, 1.0, 3.0),
        (0.0, 0.0, 1.0),
    )


def test_result_round_trip() -> None:
    result = make_result()
    restored = ClusteringResult.from_dict(result.to_dict())
    assert restored == result
