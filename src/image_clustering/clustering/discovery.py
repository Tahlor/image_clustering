"""Image discovery and sequence construction."""

from __future__ import annotations

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
