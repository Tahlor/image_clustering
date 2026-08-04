"""Tests for manifest-driven File Library cache materialization."""

import csv
from pathlib import Path

import pytest

from analysis.reviewed_real_occlusion.materialize_library_cache import (
    load_expected_basenames,
    materialize,
    normalize_basename,
)


def _manifest(path: Path, values: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path"])
        writer.writeheader()
        for value in values:
            writer.writerow({"relative_path": value})


def test_normalize_basename_removes_upload_copy_suffix() -> None:
    assert normalize_basename("images/accepted/x/image(12).JPG") == "image.JPG"


def test_materialize_copies_only_manifest_images(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    cache = tmp_path / "cache"
    output = tmp_path / "durable"
    cache.mkdir()
    _manifest(manifest, ["images/a.jpg", "images/b.jpg"])
    (cache / "a(1).jpg").write_bytes(b"a")
    (cache / "b.jpg").write_bytes(b"b")
    (cache / "generated.jpg").write_bytes(b"ignore")

    summary = materialize(manifest, [cache], output)

    assert summary.expected == 2
    assert summary.materialized == 2
    assert summary.missing == 0
    assert (output / "images_by_basename/a.jpg").read_bytes() == b"a"
    assert (output / "images_by_basename/b.jpg").read_bytes() == b"b"
    assert not (output / "images_by_basename/generated.jpg").exists()


def test_same_byte_duplicate_sources_are_allowed(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "durable"
    first.mkdir()
    second.mkdir()
    _manifest(manifest, ["images/a.jpg"])
    (first / "a.jpg").write_bytes(b"same")
    (second / "a(1).jpg").write_bytes(b"same")

    summary = materialize(manifest, [first, second], output)

    assert summary.materialized == 1
    assert summary.duplicate_sources == 1


def test_differing_byte_basename_collision_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _manifest(manifest, ["images/a.jpg"])
    (first / "a.jpg").write_bytes(b"first")
    (second / "a(1).jpg").write_bytes(b"second")

    with pytest.raises(ValueError, match="differing-byte basename collision"):
        materialize(manifest, [first, second], tmp_path / "durable")


def test_missing_manifest_rows_are_reported(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    cache = tmp_path / "cache"
    cache.mkdir()
    _manifest(manifest, ["images/a.jpg", "images/b.jpg"])
    (cache / "a.jpg").write_bytes(b"a")

    summary = materialize(manifest, [cache], tmp_path / "durable")

    assert summary.materialized == 1
    assert summary.missing == 1


def test_duplicate_manifest_basename_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    _manifest(manifest, ["accepted/one/a.jpg", "rejected/two/a.jpg"])

    with pytest.raises(ValueError, match="not globally unique"):
        load_expected_basenames(manifest)
