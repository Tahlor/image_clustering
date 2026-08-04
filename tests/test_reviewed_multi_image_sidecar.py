from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from image_clustering.evaluation.reviewed_groups import prepare_dataset
from image_clustering.evaluation.reviewed_subtypes import (
    validate_completed_subtypes,
)
from reviewed_fixture import build_package


def _completed_rows(tmp_path: Path) -> tuple[Path, Path, list[str], list[dict[str, str]]]:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    sidecar = prepared / "accepted_group_occlusion_subtypes.csv"
    with sidecar.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        row.update(
            {
                "occlusion_subtype": "same_clean",
                "visual_relationship_category": "identical_or_near_identical",
                "visual_overlay_category": "none",
                "material_occlusion_metric_included": "false",
                "affected_image_id": "",
                "affected_image_ids_json": "[]",
                "occluded_image_id": "",
                "occluded_image_ids_json": "[]",
                "better_view_image_id": "",
                "better_view_image_ids_json": "[]",
                "meaningful_hidden_content_risk": "none",
                "occlusion_size_category": "none",
                "registration_difficulty": "easy",
                "evidence": "Same physical document without hidden content.",
                "uncertainty_notes": "",
                "annotator_method": "manual full-resolution visual review",
            }
        )
    return prepared, sidecar, fieldnames, rows


def _write_rows(
    sidecar: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with sidecar.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_material_group_lists_every_occluded_and_better_view(
    tmp_path: Path,
) -> None:
    prepared, sidecar, fieldnames, rows = _completed_rows(tmp_path)
    members = json.loads(rows[0]["member_image_ids_json"])
    rows[0].update(
        {
            "occlusion_subtype": "same_occluded",
            "visual_relationship_category": "material_physical_occlusion",
            "material_occlusion_metric_included": "true",
            "affected_image_id": members[0],
            "affected_image_ids_json": json.dumps([members[0]]),
            "occluded_image_id": members[0],
            "occluded_image_ids_json": json.dumps([members[0]]),
            "better_view_image_id": members[1],
            "better_view_image_ids_json": json.dumps([members[1]]),
            "meaningful_hidden_content_risk": "high",
            "occlusion_size_category": "large",
            "registration_difficulty": "moderate",
            "evidence": "One source sheet hides indexed content visible in the other.",
        }
    )
    _write_rows(sidecar, fieldnames, rows)
    summary = validate_completed_subtypes(prepared, sidecar)
    assert summary["material_occlusion_metric_group_count"] == 1


def test_visual_image_lists_reject_nonmembers(tmp_path: Path) -> None:
    prepared, sidecar, fieldnames, rows = _completed_rows(tmp_path)
    rows[0].update(
        {
            "visual_relationship_category": "visual_only_overlay",
            "visual_overlay_category": "large_number",
            "affected_image_id": "not-a-member.jpg",
            "affected_image_ids_json": json.dumps(["not-a-member.jpg"]),
        }
    )
    _write_rows(sidecar, fieldnames, rows)
    with pytest.raises(ValueError, match="nonmember IDs"):
        validate_completed_subtypes(prepared, sidecar)
