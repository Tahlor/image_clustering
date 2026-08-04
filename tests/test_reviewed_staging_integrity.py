from __future__ import annotations

import csv
from pathlib import Path

import pytest

from image_clustering.evaluation.reviewed_groups import prepare_dataset
from reviewed_fixture import build_package


def test_staged_manifest_records_matching_source_hashes(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    stage_root = tmp_path / "staged"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
        stage_root=stage_root,
    )
    with (prepared / "evaluation_input_manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 422
    assert all(row["source_sha256"] == row["staged_sha256"] for row in rows)
    assert all(int(row["byte_count"]) > 0 for row in rows)


def test_same_size_staged_byte_mismatch_fails_closed(tmp_path: Path) -> None:
    csv_path, jsonl_path = build_package(tmp_path)
    prepared = tmp_path / "prepared"
    stage_root = tmp_path / "staged"
    prepare_dataset(
        csv_path,
        prepared,
        jsonl_path=jsonl_path,
        package_root=tmp_path,
        stage_root=stage_root,
    )
    with (prepared / "evaluation_input_manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        first = next(csv.DictReader(handle))
    target = Path(first["staged_path"])
    source = Path(first["source_path"])
    target.unlink()
    replacement = b"x" * source.stat().st_size
    if replacement == source.read_bytes():
        replacement = b"y" * source.stat().st_size
    target.write_bytes(replacement)

    with pytest.raises(ValueError, match="staged file differs from source"):
        prepare_dataset(
            csv_path,
            prepared,
            jsonl_path=jsonl_path,
            package_root=tmp_path,
            stage_root=stage_root,
        )
