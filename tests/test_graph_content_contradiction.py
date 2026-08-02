"""Graph regression tests for distributed document-content contradictions."""

from image_clustering.clustering.graph import build_clusters
from image_clustering.clustering.models import PairComparison


def _comparison(first: str, second: str, accepted: bool) -> PairComparison:
    return PairComparison(
        first_image_id=first,
        second_image_id=second,
        sequence_id=".",
        index_gap=1,
        same_document=accepted,
        confidence=0.9 if accepted else 0.1,
        reason="test",
    )


def test_distributed_ink_contradiction_blocks_transitive_bridge() -> None:
    comparisons = [
        _comparison("a.jpg", "b.jpg", True),
        _comparison("b.jpg", "c.jpg", True),
        PairComparison(
            first_image_id="a.jpg",
            second_image_id="c.jpg",
            sequence_id=".",
            index_gap=2,
            same_document=False,
            confidence=0.1,
            reason="distributed coherent ink disagreement",
            registration_model="affine",
            valid_fraction=0.95,
            feature_overlap=0.40,
            unmatched_ink_union_fraction=0.10,
            ink_mismatch_tiles_fraction=0.35,
            residual_tiles_changed_fraction=0.30,
            hard_contradiction=True,
        ),
    ]
    clusters = build_clusters(
        sequence_id=".",
        image_ids=["a.jpg", "b.jpg", "c.jpg"],
        comparisons=comparisons,
    )
    assert [cluster.image_ids for cluster in clusters] == [
        ("a.jpg", "b.jpg"),
        ("c.jpg",),
    ]


def test_rejected_multi_occlusion_pair_does_not_block_valid_bridge() -> None:
    comparisons = [
        _comparison("a.jpg", "b.jpg", True),
        _comparison("b.jpg", "c.jpg", True),
        PairComparison(
            first_image_id="a.jpg",
            second_image_id="c.jpg",
            sequence_id=".",
            index_gap=2,
            same_document=False,
            confidence=0.2,
            reason="multiple occlusion states require intermediate view",
            registration_model="homography",
            valid_fraction=0.90,
            feature_overlap=0.20,
            unmatched_ink_union_fraction=0.30,
            ink_mismatch_tiles_fraction=0.55,
            residual_tiles_changed_fraction=0.20,
            occlusion_candidate_count=2,
            occlusion_material_fraction=0.70,
            hard_contradiction=False,
        ),
    ]
    clusters = build_clusters(
        sequence_id=".",
        image_ids=["a.jpg", "b.jpg", "c.jpg"],
        comparisons=comparisons,
    )
    assert [cluster.image_ids for cluster in clusters] == [
        ("a.jpg", "b.jpg", "c.jpg"),
    ]


def test_material_occlusion_contradiction_is_bridgeable_by_two_states() -> None:
    comparisons = [
        PairComparison(
            first_image_id="a.jpg",
            second_image_id="b.jpg",
            sequence_id=".",
            index_gap=1,
            same_document=True,
            confidence=0.50,
            reason="physical occlusion",
            branch="physical_occlusion",
        ),
        PairComparison(
            first_image_id="b.jpg",
            second_image_id="c.jpg",
            sequence_id=".",
            index_gap=1,
            same_document=True,
            confidence=0.97,
            reason="physical occlusion",
            branch="physical_occlusion",
        ),
        PairComparison(
            first_image_id="a.jpg",
            second_image_id="c.jpg",
            sequence_id=".",
            index_gap=2,
            same_document=False,
            confidence=0.20,
            reason="distributed coherent ink disagreement",
            registration_model="affine",
            valid_fraction=0.93,
            feature_overlap=0.19,
            unmatched_ink_union_fraction=0.33,
            ink_mismatch_tiles_fraction=0.54,
            residual_tiles_changed_fraction=0.12,
            occlusion_candidate_count=1,
            occlusion_area_fraction=0.49,
            occlusion_residual_capture=0.63,
            occlusion_material_fraction=0.71,
            occlusion_material_median=0.065,
            hard_contradiction=True,
        ),
    ]
    clusters = build_clusters(
        sequence_id=".",
        image_ids=["a.jpg", "b.jpg", "c.jpg"],
        comparisons=comparisons,
    )
    assert [cluster.image_ids for cluster in clusters] == [
        ("a.jpg", "b.jpg", "c.jpg"),
    ]


def test_low_material_hard_negative_still_blocks_two_accepted_edges() -> None:
    comparisons = [
        PairComparison(
            first_image_id="a.jpg",
            second_image_id="b.jpg",
            sequence_id=".",
            index_gap=1,
            same_document=True,
            confidence=0.50,
            reason="physical occlusion",
            branch="physical_occlusion",
        ),
        PairComparison(
            first_image_id="b.jpg",
            second_image_id="c.jpg",
            sequence_id=".",
            index_gap=1,
            same_document=True,
            confidence=0.97,
            reason="physical occlusion",
            branch="physical_occlusion",
        ),
        PairComparison(
            first_image_id="a.jpg",
            second_image_id="c.jpg",
            sequence_id=".",
            index_gap=2,
            same_document=False,
            confidence=0.10,
            reason="different filled records",
            registration_model="affine",
            valid_fraction=0.95,
            feature_overlap=0.30,
            unmatched_ink_union_fraction=0.40,
            ink_mismatch_tiles_fraction=0.60,
            residual_tiles_changed_fraction=0.30,
            occlusion_area_fraction=0.50,
            occlusion_residual_capture=0.70,
            occlusion_material_fraction=0.10,
            occlusion_material_median=0.005,
            hard_contradiction=True,
        ),
    ]
    clusters = build_clusters(
        sequence_id=".",
        image_ids=["a.jpg", "b.jpg", "c.jpg"],
        comparisons=comparisons,
    )
    assert [cluster.image_ids for cluster in clusters] == [
        ("a.jpg",),
        ("b.jpg", "c.jpg"),
    ]
