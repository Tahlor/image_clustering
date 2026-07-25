from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_dataset import crop_artifacts, inventory


def test_inventory_records_supported_images_by_immediate_parent(tmp_path: Path) -> None:
    source = tmp_path / "input" / "sequence-a"
    source.mkdir(parents=True)
    image_path = source / "image-001.png"
    cv2.imwrite(str(image_path), np.zeros((12, 20), dtype=np.uint8))

    payload = inventory(tmp_path / "input", tmp_path / "output")

    assert payload["total_image_count"] == 1
    assert payload["immediate_parent_folder_count"] == 1
    assert payload["image_count_by_folder"] == {"sequence-a": 1}
    assert payload["decoding_failures"] == []
    assert payload["images"][0]["relative_path"] == "sequence-a/image-001.png"
    assert payload["images"][0]["width"] == 20
    assert (tmp_path / "output" / "inventory" / "dataset_inventory.csv").is_file()


def test_crop_review_includes_native_members_without_submissions(
    tmp_path: Path,
) -> None:
    source_folder = tmp_path / "input" / "sequence-a"
    output_root = tmp_path / "output"
    cluster_result = {
        "folder": "sequence-a",
        "upstream_cluster_id": "cluster_00001",
        "source_folder": str(source_folder),
        "groups": [
            {"images": ["image-001.jpg", "image-002.jpg"]},
        ],
        "submissions": [],
    }

    rows = crop_artifacts(
        {"clusters": [cluster_result]},
        tmp_path / "input",
        output_root,
        {"images": []},
    )

    assert rows == []
    page = (
        output_root
        / "review"
        / "crops"
        / "by_cluster"
        / "cluster_00001.html"
    ).read_text()
    assert page.count("Original full resolution</a>") == 2
    index = (output_root / "review" / "crops" / "index.html").read_text()
    assert "by_cluster/cluster_00001.html" in index
    assert "<td>0</td><td>0</td>" in index


def test_review_package_contains_confidence_and_tuning_controls(tmp_path: Path) -> None:
    from image_clustering.clustering.models import (
        ClusteringResult,
        ImageCluster,
        ImageItem,
        PairComparison,
    )
    from scripts.evaluate_dataset import cluster_artifacts, write_review_package

    input_root = tmp_path / "input"
    source_folder = input_root / "sequence-a"
    source_folder.mkdir(parents=True)
    first_path = source_folder / "a.png"
    second_path = source_folder / "b.png"
    cv2.imwrite(str(first_path), np.zeros((20, 30), dtype=np.uint8))
    cv2.imwrite(str(second_path), np.zeros((20, 30), dtype=np.uint8))
    result = ClusteringResult(
        config_fingerprint="fingerprint",
        input_root=input_root,
        images=(
            ImageItem("sequence-a/a.png", first_path, "sequence-a", 0),
            ImageItem("sequence-a/b.png", second_path, "sequence-a", 1),
        ),
        clusters=(
            ImageCluster(
                "cluster_00001",
                "sequence-a",
                ("sequence-a/a.png", "sequence-a/b.png"),
                "sequence-a/a.png",
            ),
        ),
        comparisons=(
            PairComparison(
                "sequence-a/a.png",
                "sequence-a/b.png",
                "sequence-a",
                1,
                True,
                0.82,
                "test",
            ),
        ),
    )
    output_root = tmp_path / "output"
    rows = cluster_artifacts(result, input_root, output_root)
    crop_rows = [{
        "submission_id": "submission_00001",
        "cluster_id": "cluster_00001",
        "source_image_path": str(first_path.resolve()),
        "source_width": 30,
        "source_height": 20,
        "bbox": [1, 2, 20, 18],
        "kind": "base_page",
        "crop_path": None,
        "source_filename": first_path.name,
        "disposition": "SUBMIT COMPLETE",
        "completeness": "complete",
        "confidence": 0.8,
        "reason": "test",
        "review_reasons": [],
    }]
    labels = tmp_path / "labels.json"
    labels.write_text(
        '{"hard_negatives": [], "near_duplicates": []}', encoding="utf-8"
    )

    write_review_package(output_root, result, rows, crop_rows, labels)
    page = (output_root / "review" / "index.html").read_text(encoding="utf-8")
    assert "confidence-asc" in page
    assert "not_a_cluster" in page
    assert "data-review-image-membership" in page
    assert "data-review-bbox" in page
    assert "localStorage" in page
    assert "export-tuning" in page
    assert (output_root / "reports" / "review_decisions.json").is_file()
