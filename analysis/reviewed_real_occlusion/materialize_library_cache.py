"""Materialize reviewed image assets from transient File Library cache paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

_IMAGE_COLUMNS = (
    "relative_path",
    "package_relative_path",
    "image_path",
    "path",
    "image_id",
    "filename",
    "source_filename",
)
_COPY_SUFFIX = re.compile(r"\(\d+\)(?=\.(?:jpe?g)$)", re.IGNORECASE)
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class MaterializationSummary:
    expected: int
    materialized: int
    missing: int
    duplicate_sources: int
    collisions: int
    output_root: str
    ledger_path: str
    missing_path: str


def normalize_basename(value: str | Path) -> str:
    """Return a manifest-compatible basename for one JPEG path."""
    basename = Path(str(value).strip().replace("\\", "/")).name
    return _COPY_SUFFIX.sub("", basename)


def _manifest_image_value(row: dict[str, str]) -> str:
    for column in _IMAGE_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            return value
    raise ValueError(
        "manifest row has no image path column; expected one of "
        + ", ".join(_IMAGE_COLUMNS)
    )


def load_expected_basenames(assignments_csv: Path) -> tuple[str, ...]:
    """Load unique normalized JPEG basenames from the authoritative manifest."""
    with assignments_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("assignments manifest is empty")
    basenames = [normalize_basename(_manifest_image_value(row)) for row in rows]
    invalid = [
        name
        for name in basenames
        if Path(name).suffix.lower() not in _IMAGE_SUFFIXES
    ]
    if invalid:
        raise ValueError(f"manifest contains non-JPEG image names: {invalid[:5]}")
    duplicates = sorted(name for name in set(basenames) if basenames.count(name) > 1)
    if duplicates:
        raise ValueError(
            "manifest basenames are not globally unique; cannot materialize "
            "by basename: "
            + ", ".join(duplicates[:10])
        )
    return tuple(basenames)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def discover_sources(
    scan_roots: Iterable[Path],
    expected: set[str],
    output_root: Path,
) -> dict[str, list[SourceFile]]:
    """Find manifest-matching JPEGs outside the durable output directory."""
    sources: dict[str, list[SourceFile]] = defaultdict(list)
    seen_paths: set[Path] = set()
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen_paths or _within(resolved, output_root):
                continue
            seen_paths.add(resolved)
            basename = normalize_basename(path.name)
            if basename not in expected:
                continue
            sources[basename].append(
                SourceFile(
                    path=resolved,
                    size=resolved.stat().st_size,
                    sha256=sha256_file(resolved),
                )
            )
    return dict(sources)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(
    assignments_csv: Path,
    scan_roots: Sequence[Path],
    output_root: Path,
    *,
    dry_run: bool = False,
) -> MaterializationSummary:
    """Copy every currently visible manifest image into one durable directory."""
    expected = load_expected_basenames(assignments_csv)
    expected_set = set(expected)
    images_dir = output_root / "images_by_basename"
    logs_dir = output_root / "logs"
    ledger_path = logs_dir / "materialized_copy_ledger.csv"
    missing_path = logs_dir / "missing_basenames.txt"
    summary_path = logs_dir / "materialization_summary.json"

    sources = discover_sources(scan_roots, expected_set, output_root)
    ledger_rows: list[dict[str, str | int]] = []
    duplicate_sources = 0
    collision_count = 0

    for basename in expected:
        candidates = sources.get(basename, [])
        hashes = {candidate.sha256 for candidate in candidates}
        if len(hashes) > 1:
            collision_count += 1
            paths = ", ".join(str(candidate.path) for candidate in candidates)
            raise ValueError(
                f"differing-byte basename collision for {basename}: {paths}"
            )
        destination = images_dir / basename
        destination_hash = sha256_file(destination) if destination.is_file() else None
        if destination_hash is not None and hashes and destination_hash not in hashes:
            collision_count += 1
            raise ValueError(
                f"durable destination differs from cache source for {basename}: "
                f"{destination}"
            )

        if candidates:
            selected = min(candidates, key=lambda item: str(item.path))
            status = (
                "already_present"
                if destination_hash == selected.sha256
                else "copied"
            )
            if not dry_run and status == "copied":
                _atomic_copy(selected.path, destination)
            ledger_rows.append(
                {
                    "normalized_basename": basename,
                    "source_path": str(selected.path),
                    "bytes": selected.size,
                    "sha256": selected.sha256,
                    "status": status,
                }
            )
            for duplicate in candidates:
                if duplicate.path == selected.path:
                    continue
                duplicate_sources += 1
                ledger_rows.append(
                    {
                        "normalized_basename": basename,
                        "source_path": str(duplicate.path),
                        "bytes": duplicate.size,
                        "sha256": duplicate.sha256,
                        "status": "duplicate_same",
                    }
                )
        elif destination_hash is not None:
            ledger_rows.append(
                {
                    "normalized_basename": basename,
                    "source_path": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": destination_hash,
                    "status": "already_present",
                }
            )

    present = {path.name for path in images_dir.glob("*") if path.is_file()}
    missing = [basename for basename in expected if basename not in present]
    summary = MaterializationSummary(
        expected=len(expected),
        materialized=len(expected) - len(missing),
        missing=len(missing),
        duplicate_sources=duplicate_sources,
        collisions=collision_count,
        output_root=str(output_root.resolve()),
        ledger_path=str(ledger_path.resolve()),
        missing_path=str(missing_path.resolve()),
    )

    if not dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "normalized_basename",
                    "source_path",
                    "bytes",
                    "sha256",
                    "status",
                ),
            )
            writer.writeheader()
            writer.writerows(ledger_rows)
        missing_path.write_text(
            "".join(f"{basename}\n" for basename in missing),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize manifest-listed reviewed JPEGs from transient File Library "
            "cache roots into a durable SHA-checked directory."
        )
    )
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat for /mnt/data and any transient per-user cache roots.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = materialize(
        assignments_csv=args.assignments_csv,
        scan_roots=args.scan_root,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return int(args.require_complete and summary.missing > 0)


if __name__ == "__main__":
    raise SystemExit(main())
