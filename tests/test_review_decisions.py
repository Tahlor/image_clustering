from pathlib import Path

import pytest

from image_clustering.review.dataset import (
    ReviewBox,
    ReviewCluster,
    ReviewDataset,
    ReviewImage,
    dataset_payload,
    review_sort_key,
)
from image_clustering.review.decisions import (
    PRIMARY_VIEW_NONE,
    approve_bboxes,
    approve_cluster,
    cluster_state,
    dissolve_cluster,
    empty_state,
    image_state,
    mark_irregular_cluster,
    mark_remaining_bboxes_ok,
    mark_remaining_clusters_ok,
    progress,
    reopen_cluster,
    restore_cluster,
    set_boxes,
    set_membership,
    set_optional,
    set_primary_view,
    set_relation,
    validate_box,
)


def build_image(
    name: str,
    index: int,
    boxes: tuple[ReviewBox, ...] = (),
) -> ReviewImage:
    return ReviewImage(
        image_id=f"folder/{name}",
        filename=name,
        source_path=Path(f"/input/folder/{name}"),
        sequence_index=index,
        width=1000,
        height=800,
        boxes=boxes,
    )


def build_dataset(
    sizes: dict[str, int],
    confidences: dict[str, float | None] | None = None,
) -> ReviewDataset:
    clusters = []
    for cluster_id, size in sizes.items():
        images = tuple(
            build_image(f"{cluster_id}-{index}.j2k", index) for index in range(size)
        )
        clusters.append(
            ReviewCluster(
                cluster_id=cluster_id,
                source_folder="folder",
                images=images,
                minimum_confidence=(confidences or {}).get(cluster_id, 0.7),
                mean_confidence=(confidences or {}).get(cluster_id, 0.7),
                largest_gap=1,
                review_reasons=(),
            )
        )
    return ReviewDataset(
        output_root=Path("/output"),
        input_root=Path("/input"),
        config_fingerprint="fingerprint",
        clusters=tuple(clusters),
    )


def test_excluding_one_member_of_a_pair_dissolves_the_cluster() -> None:
    dataset = build_dataset({"cluster_00001": 2})
    state = empty_state()
    images = dataset.cluster("cluster_00001").images

    cluster = set_membership(
        state, dataset, "cluster_00001", images[0].image_id, included=False
    )

    assert cluster["dissolved"] is True
    assert cluster["status"] == "dissolved"
    assert cluster["excluded_image_ids"] == [images[0].image_id]


def test_excluding_one_of_three_keeps_the_cluster_but_marks_it_edited() -> None:
    dataset = build_dataset({"cluster_00002": 3})
    state = empty_state()
    images = dataset.cluster("cluster_00002").images

    cluster = set_membership(
        state, dataset, "cluster_00002", images[2].image_id, included=False
    )

    assert cluster["dissolved"] is False
    assert cluster["status"] == "edited"
    assert cluster["excluded_image_ids"] == [images[2].image_id]


def test_reincluding_a_member_restores_the_cluster_and_keeps_it_reviewed() -> None:
    dataset = build_dataset({"cluster_00003": 2})
    state = empty_state()
    images = dataset.cluster("cluster_00003").images
    set_membership(state, dataset, "cluster_00003", images[0].image_id, included=False)

    cluster = set_membership(
        state, dataset, "cluster_00003", images[0].image_id, included=True
    )

    assert cluster["dissolved"] is False
    assert cluster["excluded_image_ids"] == []
    assert cluster["status"] == "edited"


def test_any_membership_change_marks_the_cluster_reviewed() -> None:
    dataset = build_dataset({"cluster_00004": 3})
    state = empty_state()
    assert cluster_state(state, "cluster_00004")["status"] == "unreviewed"

    set_membership(
        state,
        dataset,
        "cluster_00004",
        dataset.cluster("cluster_00004").images[0].image_id,
        included=False,
    )

    assert progress(state, dataset).reviewed_cluster_count == 1
    assert progress(state, dataset).unreviewed_cluster_count == 0


def test_marking_a_cluster_irregular_is_reviewed_without_dissolving_it() -> None:
    dataset = build_dataset({"cluster_irregular": 3})
    state = empty_state()

    irregular = mark_irregular_cluster(state, dataset, "cluster_irregular")

    assert irregular["status"] == "irregular"
    assert irregular["dissolved"] is False
    assert irregular["excluded_image_ids"] == []
    counts = progress(state, dataset)
    assert counts.irregular_cluster_count == 1
    assert counts.reviewed_cluster_count == 1


    dataset = build_dataset({"cluster_00005": 4})
    state = empty_state()

    dissolved = dissolve_cluster(state, dataset, "cluster_00005")
    assert dissolved["status"] == "dissolved"
    assert len(dissolved["excluded_image_ids"]) == 4

    reopened = reopen_cluster(state, dataset, "cluster_00005")
    assert reopened["status"] == "unreviewed"
    assert reopened["excluded_image_ids"] == []
    assert reopened["dissolved"] is False


def test_mark_remaining_clusters_ok_only_touches_unreviewed_clusters() -> None:
    dataset = build_dataset({"a": 2, "b": 2, "c": 3})
    state = empty_state()
    dissolve_cluster(state, dataset, "a")

    changed = mark_remaining_clusters_ok(state, dataset)

    assert sorted(changed) == ["b", "c"]
    assert cluster_state(state, "a")["status"] == "dissolved"
    assert cluster_state(state, "b")["status"] == "approved"
    counts = progress(state, dataset)
    assert counts.unreviewed_cluster_count == 0
    assert counts.approved_cluster_count == 2


def test_bbox_approval_is_independent_of_cluster_approval() -> None:
    dataset = build_dataset({"cluster_00006": 2})
    state = empty_state()
    image_id = dataset.cluster("cluster_00006").images[0].image_id

    approve_cluster(state, dataset, "cluster_00006")

    assert image_state(state, "cluster_00006", image_id)["bbox_status"] == "unreviewed"
    approve_bboxes(state, dataset, "cluster_00006", image_id)
    assert image_state(state, "cluster_00006", image_id)["bbox_status"] == "approved"
    assert cluster_state(state, "cluster_00006")["status"] == "approved"


def test_editing_boxes_marks_bboxes_edited_without_approving_the_cluster() -> None:
    dataset = build_dataset({"cluster_00007": 2})
    state = empty_state()
    image_id = dataset.cluster("cluster_00007").images[0].image_id

    record = set_boxes(
        state,
        dataset,
        "cluster_00007",
        image_id,
        [{"bbox": [10, 20, 300, 400], "kind": "reviewer"}],
    )

    assert record["bbox_status"] == "edited"
    assert record["boxes"][0]["bbox"] == [10, 20, 300, 400]
    assert cluster_state(state, "cluster_00007")["status"] == "unreviewed"


def test_a_quad_box_keeps_its_corners_and_takes_its_bbox_from_them() -> None:
    dataset = build_dataset({"cluster_00007": 2})
    state = empty_state()
    image_id = dataset.cluster("cluster_00007").images[0].image_id
    quad = [[20, 30], [300, 10], [320, 400], [5, 380]]

    record = set_boxes(
        state,
        dataset,
        "cluster_00007",
        image_id,
        [{"bbox": [0, 0, 1, 1], "polygon": quad, "kind": "base_page", "confidence": 0.9}],
    )

    box = record["boxes"][0]
    assert box["polygon"] == quad
    assert box["bbox"] == [5, 10, 320, 400]
    assert box["confidence"] == 0.9


@pytest.mark.parametrize("polygon", [[[0, 0], [10, 10]], [[0, 0], [10, 0], [-1, 5]], [[0, 0], [10, 0], ["a", 5]]])
def test_unusable_polygons_are_rejected(polygon: list) -> None:
    dataset = build_dataset({"cluster_00007": 2})
    image_id = dataset.cluster("cluster_00007").images[0].image_id
    with pytest.raises(ValueError):
        set_boxes(empty_state(), dataset, "cluster_00007", image_id, [{"bbox": [0, 0, 5, 5], "polygon": polygon}])


def test_mark_remaining_bboxes_ok_preserves_edited_state() -> None:
    dataset = build_dataset({"cluster_00008": 2})
    state = empty_state()
    images = dataset.cluster("cluster_00008").images
    set_boxes(
        state,
        dataset,
        "cluster_00008",
        images[0].image_id,
        [{"bbox": [1, 1, 50, 50]}],
    )

    changed = mark_remaining_bboxes_ok(state, dataset)

    edited = image_state(state, "cluster_00008", images[0].image_id)
    approved = image_state(state, "cluster_00008", images[1].image_id)
    assert changed == [images[1].image_id]
    assert edited["bbox_status"] == "edited"
    assert approved["bbox_status"] == "approved"


@pytest.mark.parametrize(
    "box",
    [
        [10, 10, 10, 50],
        [-1, 0, 20, 20],
        [0, 0, 2000, 20],
        [0, 0, 20, 2000],
        [0, 0, 20],
    ],
)
def test_validate_box_rejects_unusable_geometry(box: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_box(box, 1000, 800)


def test_validate_box_accepts_boxes_inside_the_source() -> None:
    assert validate_box([0, 0, 1000, 800], 1000, 800) == (0, 0, 1000, 800)


def test_review_queue_defaults_to_weakest_confidence_first() -> None:
    dataset = build_dataset(
        {"strong": 2, "weak": 2, "unregistered": 1},
        {"strong": 0.95, "weak": 0.41, "unregistered": None},
    )
    ordered = sorted(dataset.clusters, key=review_sort_key)

    assert [cluster.cluster_id for cluster in ordered] == [
        "weak",
        "strong",
        "unregistered",
    ]
    payload = dataset_payload(dataset)
    assert payload["defaults"] == {"minimum_cluster_size": 2, "sort": "confidence-asc"}


def test_restore_cluster_undoes_a_dissolve() -> None:
    dataset = build_dataset({"cluster_00009": 2})
    state = empty_state()
    before = {
        key: value
        for key, value in cluster_state(state, "cluster_00009").items()
        if key != "updated_at"
    }
    dissolve_cluster(state, dataset, "cluster_00009")

    restored = restore_cluster(state, dataset, "cluster_00009", before)

    assert restored["status"] == "unreviewed"
    assert restored["dissolved"] is False
    assert restored["excluded_image_ids"] == []
    assert progress(state, dataset).unreviewed_cluster_count == 1


def test_restore_cluster_undoes_a_box_edit() -> None:
    dataset = build_dataset({"cluster_00010": 2})
    state = empty_state()
    image_id = dataset.cluster("cluster_00010").images[0].image_id
    snapshot = {
        "status": "unreviewed",
        "dissolved": False,
        "excluded_image_ids": [],
        "images": {},
    }
    set_boxes(state, dataset, "cluster_00010", image_id, [{"bbox": [3, 4, 90, 90]}])
    assert image_state(state, "cluster_00010", image_id)["boxes"] is not None

    restore_cluster(state, dataset, "cluster_00010", snapshot)

    assert image_state(state, "cluster_00010", image_id)["boxes"] is None
    assert image_state(state, "cluster_00010", image_id)["bbox_status"] == "unreviewed"


def test_restore_cluster_rejects_unknown_status_and_invalid_boxes() -> None:
    dataset = build_dataset({"cluster_00011": 2})
    state = empty_state()
    image_id = dataset.cluster("cluster_00011").images[0].image_id

    with pytest.raises(ValueError):
        restore_cluster(state, dataset, "cluster_00011", {"status": "bogus"})
    with pytest.raises(ValueError):
        restore_cluster(
            state,
            dataset,
            "cluster_00011",
            {
                "status": "edited",
                "images": {image_id: {"boxes": [{"bbox": [0, 0, 9999, 10]}]}},
            },
        )


def test_restore_cluster_drops_unknown_images_and_exclusions() -> None:
    dataset = build_dataset({"cluster_00012": 3})
    state = empty_state()
    known = dataset.cluster("cluster_00012").images[0].image_id

    restored = restore_cluster(
        state,
        dataset,
        "cluster_00012",
        {
            "status": "edited",
            "excluded_image_ids": [known, "folder/not-a-member.j2k"],
            "images": {
                known: {"included": False},
                "folder/not-a-member.j2k": {"included": False},
            },
        },
    )

    assert restored["excluded_image_ids"] == [known]
    assert set(restored["images"]) == {
        image.image_id for image in dataset.cluster("cluster_00012").images
    }


def test_relation_sets_the_implied_membership_status() -> None:
    dataset = build_dataset({"same": 2, "background": 2, "different": 2, "unsure": 2})
    state = empty_state()

    assert set_relation(state, dataset, "same", "same_document")["status"] == "approved"
    assert (
        set_relation(state, dataset, "background", "same_background_document")["status"]
        == "approved"
    )
    different = set_relation(state, dataset, "different", "different_document")
    assert different["status"] == "dissolved"
    assert different["dissolved"] is True
    assert set_relation(state, dataset, "unsure", "unsure")["status"] == "irregular"

    counts = progress(state, dataset).relation_counts
    assert counts == {
        "same_document": 1,
        "same_background_document": 1,
        "different_document": 1,
        "unsure": 1,
    }


def test_relation_rejects_unknown_values() -> None:
    dataset = build_dataset({"cluster_00020": 2})
    with pytest.raises(ValueError):
        set_relation(empty_state(), dataset, "cluster_00020", "same")


def test_contradicting_status_change_clears_the_relation() -> None:
    dataset = build_dataset({"cluster_00021": 2, "cluster_00022": 3})
    state = empty_state()
    set_relation(state, dataset, "cluster_00021", "same_background_document")

    assert dissolve_cluster(state, dataset, "cluster_00021")["relation"] is None

    set_relation(state, dataset, "cluster_00021", "different_document")
    assert approve_cluster(state, dataset, "cluster_00021")["relation"] is None

    # Editing membership of a confirmed 3-capture cluster keeps a same-* relation.
    set_relation(state, dataset, "cluster_00022", "same_document")
    image_id = dataset.cluster("cluster_00022").images[2].image_id
    edited = set_membership(state, dataset, "cluster_00022", image_id, False)
    assert edited["status"] == "edited"
    assert edited["relation"] == "same_document"


def test_primary_view_accepts_members_or_none_and_survives_relation_changes() -> None:
    dataset = build_dataset({"cluster_00023": 2})
    state = empty_state()
    first = dataset.cluster("cluster_00023").images[0].image_id

    set_relation(state, dataset, "cluster_00023", "same_document")
    assert set_primary_view(state, dataset, "cluster_00023", first)["primary_view"] == first
    assert set_relation(state, dataset, "cluster_00023", "same_background_document")[
        "primary_view"
    ] == first
    assert (
        set_primary_view(state, dataset, "cluster_00023", PRIMARY_VIEW_NONE)["primary_view"]
        == "none"
    )
    assert progress(state, dataset).primary_view_set_count == 1
    with pytest.raises(ValueError):
        set_primary_view(state, dataset, "cluster_00023", "folder/not-a-member.j2k")


def test_reopen_clears_relation_and_primary_view() -> None:
    dataset = build_dataset({"cluster_00024": 2})
    state = empty_state()
    first = dataset.cluster("cluster_00024").images[0].image_id
    set_relation(state, dataset, "cluster_00024", "same_document")
    set_primary_view(state, dataset, "cluster_00024", first)

    reopened = reopen_cluster(state, dataset, "cluster_00024")

    assert reopened["relation"] is None
    assert reopened["primary_view"] is None


def test_restore_round_trips_relation_and_validates_it() -> None:
    dataset = build_dataset({"cluster_00025": 2})
    state = empty_state()
    first = dataset.cluster("cluster_00025").images[0].image_id
    set_relation(state, dataset, "cluster_00025", "same_background_document")
    set_primary_view(state, dataset, "cluster_00025", first)
    snapshot = {
        key: value
        for key, value in cluster_state(state, "cluster_00025").items()
        if key != "updated_at"
    }
    reopen_cluster(state, dataset, "cluster_00025")

    restored = restore_cluster(state, dataset, "cluster_00025", snapshot)

    assert restored["relation"] == "same_background_document"
    assert restored["primary_view"] == first
    with pytest.raises(ValueError):
        restore_cluster(
            state, dataset, "cluster_00025", {**snapshot, "relation": "bogus"}
        )
    with pytest.raises(ValueError):
        restore_cluster(
            state,
            dataset,
            "cluster_00025",
            {**snapshot, "primary_view": "folder/not-a-member.j2k"},
        )
    # A restored relation that contradicts the restored status is dropped.
    contradictory = restore_cluster(
        state, dataset, "cluster_00025", {**snapshot, "status": "dissolved"}
    )
    assert contradictory["relation"] is None


def test_legacy_decision_records_gain_empty_relation_fields() -> None:
    state = {"clusters": {"cluster_00026": {"status": "approved"}}}
    record = cluster_state(state, "cluster_00026")
    assert record["relation"] is None
    assert record["primary_view"] is None


def test_optional_flag_is_independent_of_the_decision_and_cleared_by_reopen() -> None:
    dataset = build_dataset({"cluster_00007": 2})
    state = empty_state()
    set_relation(state, dataset, "cluster_00007", "same_background_document")
    record = set_optional(state, dataset, "cluster_00007", True)
    assert record["optional"] is True
    assert record["relation"] == "same_background_document"
    assert progress(state, dataset).optional_cluster_count == 1
    snapshot = dict(cluster_state(state, "cluster_00007"))
    reopen_cluster(state, dataset, "cluster_00007")
    assert cluster_state(state, "cluster_00007")["optional"] is False
    restore_cluster(state, dataset, "cluster_00007", snapshot)
    assert cluster_state(state, "cluster_00007")["optional"] is True


def test_an_optional_box_keeps_its_flag() -> None:
    dataset = build_dataset({"cluster_00007": 2})
    image_id = dataset.cluster("cluster_00007").images[0].image_id
    record = set_boxes(empty_state(), dataset, "cluster_00007", image_id,
                       [{"bbox": [10, 20, 300, 400], "optional": True}, {"bbox": [10, 20, 30, 40]}])
    assert record["boxes"][0]["optional"] is True
    assert "optional" not in record["boxes"][1]
