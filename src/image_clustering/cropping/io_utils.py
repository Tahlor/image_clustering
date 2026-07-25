from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from .models import BBox, ImageRecord


def discover_sequences(
    input_dir: Path, suffixes: Iterable[str]
) -> list[tuple[Path, list[ImageRecord]]]:
    """Discover filename-sorted image sequences by immediate parent folder.

    Args:
        input_dir: Root input directory.
        suffixes: Supported lowercase suffixes.

    Returns:
        One ordered sequence per immediate image-containing folder.
    """
    suffix_set = {suffix.lower() for suffix in suffixes}
    grouped: dict[Path, list[Path]] = {}
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffix_set:
            grouped.setdefault(path.parent, []).append(path)

    sequences: list[tuple[Path, list[ImageRecord]]] = []
    global_index = 0
    for folder in sorted(
        grouped, key=lambda item: item.relative_to(input_dir).as_posix()
    ):
        paths = sorted(grouped[folder], key=lambda item: item.name)
        records = []
        for path in paths:
            records.append(
                ImageRecord(
                    index=global_index,
                    path=path,
                    folder=folder,
                    name=path.name,
                )
            )
            global_index += 1
        sequences.append((folder, records))
    return sequences


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {path}")
    return image


def resize_max(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / float(max(height, width)))
    if scale >= 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(
        image,
        (max(2, round(width * scale)), max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def clip_bbox(bbox: BBox, shape: tuple[int, int]) -> BBox:
    height, width = shape
    x0, y0, x1, y1 = bbox
    return (
        max(0, min(width, int(x0))),
        max(0, min(height, int(y0))),
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
    )


def crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
    x0, y0, x1, y1 = clip_bbox(bbox, image.shape[:2])
    return image[y0:y1, x0:x1]


def write_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Failed to write {path}")


def read_json(path: Path) -> object:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
