from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from image_clustering.evaluation.reviewed_groups import (
    evaluate_predictions,
    fit_real_calibration,
    prepare_dataset,
)
from image_clustering.evaluation.reviewed_prepare import load_subtypes
from image_clustering.evaluation.reviewed_subtypes import (
    REQUIRED_COMPLETED_FIELDS,
    validate_completed_subtypes,
)
from reviewed_fixture import build_package, write_predictions


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_completed_sidecar(tmp_path: Path) -> tuple[Path, Path]:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    sidecar = prepared / "accepted_group_occlusion_subtypes.csv"
    fieldnames, rows = _read_rows(sidecar)
    for row in rows:
        row.update(
            {
                "occlusion_subtype": "same_clean",
                "visual_relationship_category": "identical_or_near_identical",
                "visual_overlay_category": "none",
                "material_occlusion_metric_included": "false",
                "affected_image_id": "",
                "occluded_image_id": "",
                "better_view_image_id": "",
                "meaningful_hidden_content_risk": "none",
                "occlusion_size_category": "none",
                "registration_difficulty": "easy",
                "evidence": "Same physical document without hidden content.",
                "uncertainty_notes": "",
                "annotator_method": "manual full-resolution visual review",
            }
        )
    _write_rows(sidecar, fieldnames, rows)
    return prepared, sidecar


def _write_pair_results(prepared: Path, path: Path) -> None:
    predictions = path.parent / "predictions.jsonl"
    write_predictions(prepared, predictions)
    evaluate_predictions(prepared, predictions, path.parent / "evaluation")
    source = path.parent / "evaluation" / "pair_results.jsonl"
    path.write_bytes(source.read_bytes())


def test_template_has_explicit_visual_evidence_fields(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    sidecar = prepared / "accepted_group_occlusion_subtypes.csv"
    fieldnames, rows = _read_rows(sidecar)
    assert REQUIRED_COMPLETED_FIELDS <= set(fieldnames)
    assert len(rows) == 134
    canonical = {
        row["original_cluster_id"]: row["image_ids"]
        for row in (
            json.loads(line)
            for line in (
                prepared / "canonical_reviewed_groups.jsonl"
            ).read_text().splitlines()
        )
        if row["review_decision"] == "accepted"
    }
    for row in rows:
        assert json.loads(row["member_image_ids_json"]) == canonical[
            row["original_cluster_id"]
        ]


def test_completed_sidecar_passes_exact_group_validation(tmp_path: Path) -> None:
    prepared, sidecar = _prepare_completed_sidecar(tmp_path)
    summary = validate_completed_subtypes(prepared, sidecar)
    assert summary["complete"]
    assert summary["accepted_group_count"] == 134
    assert summary["material_occlusion_metric_group_count"] == 0


def test_visual_only_overlay_cannot_enter_material_metric(tmp_path: Path) -> None:
    prepared, sidecar = _prepare_completed_sidecar(tmp_path)
    fieldnames, rows = _read_rows(sidecar)
    members = json.loads(rows[0]["member_image_ids_json"])
    rows[0].update(
        {
            "visual_relationship_category": "visual_only_overlay",
            "visual_overlay_category": "large_number",
            "material_occlusion_metric_included": "true",
            "affected_image_id": members[0],
            "occluded_image_id": members[0],
            "better_view_image_id": members[1],
            "meaningful_hidden_content_risk": "low",
            "occlusion_size_category": "medium",
        }
    )
    _write_rows(sidecar, fieldnames, rows)
    with pytest.raises(ValueError, match="incompatible relationship"):
        validate_completed_subtypes(prepared, sidecar)


def test_member_image_ids_must_match_authority(tmp_path: Path) -> None:
    prepared, sidecar = _prepare_completed_sidecar(tmp_path)
    fieldnames, rows = _read_rows(sidecar)
    rows[0]["member_image_ids_json"] = json.dumps(["not-in-authority.jpg"])
    _write_rows(sidecar, fieldnames, rows)
    with pytest.raises(ValueError, match="differs from authority"):
        validate_completed_subtypes(prepared, sidecar)


def test_duplicate_subtype_rows_fail_closed(tmp_path: Path) -> None:
    prepared, sidecar = _prepare_completed_sidecar(tmp_path)
    fieldnames, rows = _read_rows(sidecar)
    rows.append(dict(rows[0]))
    _write_rows(sidecar, fieldnames, rows)
    with pytest.raises(ValueError, match="Duplicate subtype row"):
        load_subtypes(sidecar)


def test_explicit_sidecar_must_be_complete_before_calibration(
    tmp_path: Path,
) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
    )
    pair_results = tmp_path / "pair_results.jsonl"
    _write_pair_results(prepared, pair_results)
    sidecar = prepared / "accepted_group_occlusion_subtypes.csv"
    with pytest.raises(ValueError, match="Unresolved occlusion subtype"):
        fit_real_calibration(
            prepared,
            pair_results,
            tmp_path / "calibrator.json",
            subtype_path=sidecar,
        )


def test_completed_sidecar_drives_material_occlusion_calibration(
    tmp_path: Path,
) -> None:
    prepared, sidecar = _prepare_completed_sidecar(tmp_path)
    pair_results = tmp_path / "pair_results.jsonl"
    _write_pair_results(prepared, pair_results)
    calibrator = fit_real_calibration(
        prepared,
        pair_results,
        tmp_path / "calibrator.json",
        subtype_path=sidecar,
    )
    assert calibrator["conditional_occlusion_truth"] == (
        "material_occlusion_metric_included"
    )
    assert calibrator["conditional_occlusion_isotonic_knots"] is not None
    assert calibrator["subtype_validation"]["accepted_group_count"] == 134
