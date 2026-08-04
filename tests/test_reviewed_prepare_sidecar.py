from __future__ import annotations

import csv
from pathlib import Path

from image_clustering.evaluation.reviewed_groups import prepare_dataset
from reviewed_fixture import build_package


def test_prepare_rerun_preserves_existing_visual_sidecar(
    tmp_path: Path,
) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    first = prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    assert first["subtype_sidecar_status"] == "created_from_template"

    sidecar = prepared / "accepted_group_occlusion_subtypes.csv"
    with sidecar.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0]["evidence"] = "human-reviewed marker"
    with sidecar.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    expected_bytes = sidecar.read_bytes()

    second = prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    assert second["subtype_sidecar_status"] == "preserved_existing"
    assert sidecar.read_bytes() == expected_bytes

    template = prepared / "accepted_group_occlusion_subtypes_template.csv"
    assert template.exists()
    assert b"human-reviewed marker" not in template.read_bytes()
