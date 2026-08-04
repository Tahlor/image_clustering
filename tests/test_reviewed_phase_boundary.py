from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from image_clustering.evaluation.reviewed_groups import (
    evaluate_predictions,
    prepare_dataset,
)
from reviewed_fixture import build_package, write_predictions

PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "reviewed_real_occlusion"
    / "run_pipeline.py"
)


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "reviewed_real_run_pipeline",
        PIPELINE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_pipeline_phase_excludes_locked_audit() -> None:
    pipeline = _load_pipeline_module()
    args = pipeline.parse_args(
        [
            "--package-root",
            "/tmp/package",
            "--assignments-csv",
            "/tmp/assignments.csv",
            "--assignments-jsonl",
            "/tmp/assignments.jsonl",
            "--output-root",
            "/tmp/output",
            "--config",
            "/tmp/config.json",
        ]
    )
    assert args.phase == "baseline"
    assert pipeline.TUNING_SPLITS == {"development", "selection"}
    assert "locked_audit" not in pipeline.TUNING_SPLITS


def test_pipeline_group_filter_keeps_audit_out_of_tuning() -> None:
    pipeline = _load_pipeline_module()
    groups = {
        "dev": [{"image_id": "a"}],
        "selection": [{"image_id": "b"}],
        "audit": [{"image_id": "c"}],
    }
    assignments = {
        "dev": "development",
        "selection": "selection",
        "audit": "locked_audit",
    }
    selected = pipeline._groups_for_splits(
        groups,
        assignments,
        pipeline.TUNING_SPLITS,
    )
    assert set(selected) == {"dev", "selection"}


def test_evaluation_split_filter_does_not_emit_locked_rows(
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
    predictions = tmp_path / "predictions.jsonl"
    write_predictions(prepared, predictions)
    output = tmp_path / "result"
    metrics = evaluate_predictions(
        prepared,
        predictions,
        output,
        include_splits={"development", "selection"},
    )
    assert metrics["evaluated_splits"] == ["development", "selection"]
    assert metrics["locked_audit_included"] is False
    assert "locked_audit" not in metrics["by_split"]
    pair_rows = [
        json.loads(line)
        for line in (output / "pair_results.jsonl").read_text().splitlines()
    ]
    group_rows = [
        json.loads(line)
        for line in (output / "group_results.jsonl").read_text().splitlines()
    ]
    assert pair_rows and group_rows
    assert {row["split"] for row in pair_rows} == {
        "development",
        "selection",
    }
    assert {row["split"] for row in group_rows} == {
        "development",
        "selection",
    }
