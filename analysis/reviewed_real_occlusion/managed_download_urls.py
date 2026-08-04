"""Build exact broker URLs for missing reviewed-real Vermont images.

This module performs no network requests. It validates every object identity against
``assignments.csv`` and prints token-bearing URLs only to stdout for the platform-
managed downloader. Tokens and signed redirects must never be written to receipts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence
from urllib.parse import quote, urlencode

ALLOWED_PREFIX = "data/vermont_naturalization"
DEFAULT_ENDPOINT = "https://dcj2khs11f.execute-api.us-east-1.amazonaws.com"
DEFAULT_TOKEN_ENV = "VERMONT_IMAGE_ACCESS_TOKEN"
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


class DownloadPlanError(RuntimeError):
    """Raised when the authoritative manifest cannot produce a safe exact key."""


@dataclass(frozen=True)
class ManagedObject:
    index: int
    normalized_basename: str
    image_id: str
    package_relative_path: str
    object_key: str
    destination: str


def _safe_parts(value: str, *, field: str) -> tuple[str, ...]:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts
    if (
        not normalized
        or normalized.startswith("/")
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise DownloadPlanError(f"unsafe {field}: {value!r}")
    return parts


def _manifest_objects(assignments_csv: Path, output_root: Path) -> list[ManagedObject]:
    with assignments_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DownloadPlanError("assignments manifest is empty")

    objects: list[ManagedObject] = []
    seen_basenames: set[str] = set()
    for position, row in enumerate(rows):
        project = (row.get("source_project") or "").strip()
        sequence_id = (row.get("sequence_id") or "").strip().replace("\\", "/")
        image_id = (row.get("image_id") or "").strip().replace("\\", "/")
        package_path = (row.get("package_relative_path") or "").strip()

        project_parts = _safe_parts(project, field="source_project")
        sequence_parts = _safe_parts(sequence_id, field="sequence_id")
        image_parts = _safe_parts(image_id, field="image_id")
        package_parts = _safe_parts(package_path, field="package_relative_path")
        if len(project_parts) != 1:
            raise DownloadPlanError(f"source_project must be one segment: {project!r}")
        if len(sequence_parts) != 2 or sequence_parts[0] != project:
            raise DownloadPlanError(
                f"sequence_id is inconsistent with source_project: {sequence_id!r}"
            )
        if len(image_parts) != 3 or image_parts[:2] != sequence_parts:
            raise DownloadPlanError(
                f"image_id is inconsistent with sequence_id: {image_id!r}"
            )
        if tuple(package_parts[-3:]) != image_parts:
            raise DownloadPlanError(
                "package_relative_path does not end with the exact image_id: "
                f"{package_path!r}"
            )
        basename = image_parts[-1]
        if Path(basename).suffix.lower() not in _IMAGE_SUFFIXES:
            raise DownloadPlanError(f"non-JPEG reviewed image: {image_id!r}")
        if basename in seen_basenames:
            raise DownloadPlanError(
                f"manifest basename is not globally unique: {basename!r}"
            )
        seen_basenames.add(basename)
        key = f"{ALLOWED_PREFIX}/{image_id}"
        destination = output_root / "images_by_basename" / basename
        objects.append(
            ManagedObject(
                index=position,
                normalized_basename=basename,
                image_id=image_id,
                package_relative_path=package_path,
                object_key=key,
                destination=str(destination.resolve()),
            )
        )
    return objects


def load_managed_objects(
    assignments_csv: Path,
    output_root: Path,
    *,
    missing_only: bool = True,
) -> tuple[ManagedObject, ...]:
    """Return exact manifest-backed objects, optionally excluding present outputs."""
    objects = _manifest_objects(assignments_csv, output_root)
    if missing_only:
        objects = [item for item in objects if not Path(item.destination).is_file()]
    return tuple(objects)


def object_request_url(object_key: str, access_token: str) -> str:
    """Return the fixed-endpoint request URL for one exact authorized object key."""
    if not access_token:
        raise DownloadPlanError("access token is empty")
    if not object_key.startswith(ALLOWED_PREFIX + "/"):
        raise DownloadPlanError("object key is outside the authorized prefix")
    encoded_key = quote(object_key, safe="")
    query = urlencode({"access": access_token})
    return f"{DEFAULT_ENDPOINT}/object/{encoded_key}?{query}"


def request_rows(
    assignments_csv: Path,
    output_root: Path,
    access_token: str,
    *,
    missing_only: bool = True,
) -> list[dict[str, object]]:
    """Build downloader rows without performing I/O or persisting the token."""
    rows: list[dict[str, object]] = []
    for item in load_managed_objects(
        assignments_csv,
        output_root,
        missing_only=missing_only,
    ):
        payload = asdict(item)
        payload["url"] = object_request_url(item.object_key, access_token)
        rows.append(payload)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV)
    parser.add_argument(
        "--include-present",
        action="store_true",
        help="Print all manifest objects instead of only missing destinations.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise DownloadPlanError(
            f"Set {args.token_env}; the token is read only for stdout URL output"
        )
    rows = request_rows(
        args.assignments_csv,
        args.output_root,
        token,
        missing_only=not args.include_present,
    )
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
