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
