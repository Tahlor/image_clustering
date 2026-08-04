"""Tests for exact managed-download URL planning of reviewed Vermont images."""

import csv
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pytest

from analysis.reviewed_real_occlusion.managed_download_urls import (
    ALLOWED_PREFIX,
    DEFAULT_ENDPOINT,
    DownloadPlanError,
    load_managed_objects,
    object_request_url,
    request_rows,
)


def _manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "image_id",
        "source_project",
        "sequence_id",
        "package_relative_path",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(filename: str = "image one.jpg") -> dict[str, str]:
    image_id = f"63129.IMG.001/media-uuid/{filename}"
    return {
        "image_id": image_id,
        "source_project": "63129.IMG.001",
        "sequence_id": "63129.IMG.001/media-uuid",
        "package_relative_path": f"images/accepted/cluster_1/{image_id}",
    }


def test_load_managed_objects_builds_exact_authorized_key(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    output = tmp_path / "output"
    _manifest(manifest, [_row()])

    objects = load_managed_objects(manifest, output)

    assert len(objects) == 1
    assert objects[0].object_key == (
        f"{ALLOWED_PREFIX}/63129.IMG.001/media-uuid/image one.jpg"
    )
    assert objects[0].destination.endswith("images_by_basename/image one.jpg")


def test_request_url_encodes_the_complete_key_and_token() -> None:
    key = f"{ALLOWED_PREFIX}/63129.IMG.001/media-uuid/image one.jpg"

    url = object_request_url(key, "a token/+?")
    parsed = urlparse(url)

    assert f"{parsed.scheme}://{parsed.netloc}" == DEFAULT_ENDPOINT
    encoded_key = parsed.path.removeprefix("/object/")
    assert unquote(encoded_key) == key
    assert "%2F" in encoded_key
    assert parse_qs(parsed.query) == {"access": ["a token/+?"]}


def test_missing_only_skips_existing_durable_image(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    output = tmp_path / "output"
    _manifest(manifest, [_row("present.jpg"), _row("missing.jpg")])
    present = output / "images_by_basename/present.jpg"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"jpeg")

    rows = request_rows(manifest, output, "token")

    assert [row["normalized_basename"] for row in rows] == ["missing.jpg"]


def test_manifest_identity_disagreement_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    row = _row()
    row["image_id"] = "63129.IMG.001/other-media/image one.jpg"
    _manifest(manifest, [row])

    with pytest.raises(DownloadPlanError, match="inconsistent with sequence_id"):
        load_managed_objects(manifest, tmp_path / "output")


def test_package_path_must_end_with_exact_image_id(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    row = _row()
    row["package_relative_path"] = "images/accepted/cluster_1/other.jpg"
    _manifest(manifest, [row])

    with pytest.raises(DownloadPlanError, match="exact image_id"):
        load_managed_objects(manifest, tmp_path / "output")


def test_duplicate_basename_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "assignments.csv"
    first = _row("same.jpg")
    second = {
        "image_id": "63129.IMG.002/other-media/same.jpg",
        "source_project": "63129.IMG.002",
        "sequence_id": "63129.IMG.002/other-media",
        "package_relative_path": (
            "images/accepted/cluster_2/63129.IMG.002/other-media/same.jpg"
        ),
    }
    _manifest(manifest, [first, second])

    with pytest.raises(DownloadPlanError, match="not globally unique"):
        load_managed_objects(manifest, tmp_path / "output")


def test_rejects_keys_outside_the_authorized_prefix() -> None:
    with pytest.raises(DownloadPlanError, match="outside the authorized prefix"):
        object_request_url("other-prefix/file.jpg", "token")
