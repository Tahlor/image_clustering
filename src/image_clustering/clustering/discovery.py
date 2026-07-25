"""Image discovery and sequence construction."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from image_clustering.clustering.models import ImageItem

SUPPORTED_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".j2k", ".jp2"}
)


def image_id(sequence_id: str, filename: str) -> str:
    """Build a stable, root-relative image identifier."""
    return filename if sequence_id == "." else f"{sequence_id}/{filename}"


def make_image_items(
    image_paths: Sequence[Path],
    sequence_id: str = ".",
) -> tuple[ImageItem, ...]:
    """Build ordered image items from an explicitly ordered path sequence."""
    paths = tuple(Path(path).resolve() for path in image_paths)
    if not paths:
        raise ValueError("image_paths cannot be empty")
    filenames = [path.name for path in paths]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Image filenames must be unique within a sequence")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Image files do not exist: {missing}")
    unsupported = [
        path for path in paths if path.suffix.lower() not in SUPPORTED_SUFFIXES
    ]
    if unsupported:
        raise ValueError(f"Unsupported image suffixes: {unsupported}")
    return tuple(
        ImageItem(
            image_id=image_id(sequence_id=sequence_id, filename=path.name),
            path=path,
            sequence_id=sequence_id,
            sequence_index=index,
        )
        for index, path in enumerate(paths)
    )


def discover_sequences(input_dir: Path) -> tuple[tuple[ImageItem, ...], ...]:
    """Discover one filename-ordered image sequence per parent folder.

    Images are never compared across parent folders.
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    grouped: dict[Path, list[Path]] = {}
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            grouped.setdefault(path.parent, []).append(path)
    folders = sorted(
        grouped,
        key=lambda folder: folder.relative_to(input_dir).as_posix(),
    )
    sequences = []
    for folder in folders:
        sequence_id = folder.relative_to(input_dir).as_posix() or "."
        ordered_paths = sorted(grouped[folder], key=lambda path: path.name)
        sequences.append(
            make_image_items(image_paths=ordered_paths, sequence_id=sequence_id)
        )
    if not sequences:
        raise FileNotFoundError(f"No supported images found under {input_dir}")
    return tuple(sequences)


def discover_triplet_sequences(
    input_dir: Path,
    manifest_path: Path,
    max_group_size: int = 3,
) -> tuple[tuple[ImageItem, ...], ...]:
    """Build independent ordered sequences from a neighbor-triplet manifest.

    Each manifest group is identified by ``source_sample_row``, ``neighbor_of``,
    and ``media_item_id``. Only files listed in the manifest are returned, so
    sibling preview conversions or unlisted images cannot enter the run. A
    group is intentionally kept separate from every other group even when its
    images are adjacent in the source sequence.
    """
    input_dir = input_dir.resolve()
    manifest_path = manifest_path.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Triplet manifest does not exist: {manifest_path}")
    if max_group_size < 1:
        raise ValueError("max_group_size must be at least 1")

    available: dict[str, Path] = {}
    by_parent_and_name: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            relative = path.relative_to(input_dir).as_posix()
            available[relative] = path.resolve()
            by_parent_and_name[(path.parent.name, path.name)].append(path.resolve())

    required = {"source_sample_row", "neighbor_of", "media_item_id", "filename"}
    groups: dict[tuple[str, str, str], list[tuple[int, str, Path]]] = defaultdict(list)
    used_paths: dict[Path, tuple[str, str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"Triplet manifest is missing required columns: {missing_columns}"
            )
        for row_number, row in enumerate(reader, start=2):
            key = (
                str(row["source_sample_row"]).strip(),
                str(row["neighbor_of"]).strip(),
                str(row["media_item_id"]).strip(),
            )
            if not all(key):
                raise ValueError(
                    f"Triplet manifest row {row_number} has an empty group key"
                )
            filename = str(row["filename"]).strip()
            if not filename:
                raise ValueError(f"Triplet manifest row {row_number} has no filename")
            relative_value = (
                str(row.get("relative_path") or "").strip().replace("\\", "/")
            )
            candidate = available.get(relative_value) if relative_value else None
            if candidate is None:
                candidate = available.get(f"{key[2]}/{Path(filename).name}")
            if candidate is None:
                matches = by_parent_and_name.get((key[2], Path(filename).name), [])
                if len(matches) == 1:
                    candidate = matches[0]
            if candidate is None:
                raise FileNotFoundError(
                    f"Triplet manifest row {row_number} references missing image "
                    f"{key[2]}/{filename} under {input_dir}"
                )
            previous_group = used_paths.get(candidate)
            if previous_group is not None and previous_group != key:
                raise ValueError(
                    f"Image {candidate} belongs to multiple triplet groups: "
                    f"{previous_group} and {key}"
                )
            used_paths[candidate] = key
            try:
                sequence_index = int(row.get("sequence_index", len(groups[key])))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Triplet manifest row {row_number} has invalid sequence_index"
                ) from error
            relation = str(row.get("relation") or "").strip()
            groups[key].append((sequence_index, relation, candidate))

    sequences: list[tuple[ImageItem, ...]] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            (0, int(item[0][0])) if item[0][0].isdigit() else (1, item[0][0]),
            item[0][2],
            item[0][1],
        ),
    )
    for key, entries in ordered_groups:
        if len(entries) > max_group_size:
            raise ValueError(
                f"Triplet group {key} contains {len(entries)} images; "
                f"maximum is {max_group_size}"
            )
        entries.sort(key=lambda entry: (entry[0], entry[2].name))
        if len({entry[2] for entry in entries}) != len(entries):
            raise ValueError(f"Triplet group {key} contains a duplicate image")
        sequence_id = f"triplet/{key[2]}/{key[0]}"
        sequences.append(
            make_image_items(
                image_paths=[entry[2] for entry in entries],
                sequence_id=sequence_id,
            )
        )
    if not sequences:
        raise FileNotFoundError(
            f"Triplet manifest contains no image groups: {manifest_path}"
        )
    return tuple(sequences)
