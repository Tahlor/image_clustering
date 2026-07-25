"""Clean, browser-readable previews for reviewer bbox editing.

The cropper's annotated JPEGs already have boxes drawn on them, so they cannot
be used to edit boxes. These previews are unannotated renderings of the original
source captures, cached on first request and reused afterwards.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2

LOGGER = logging.getLogger(__name__)

EDIT_MAX_DIMENSION = 1600
THUMBNAIL_MAX_DIMENSION = 480
JPEG_QUALITY = 88


def preview_cache_dir(output_root: Path) -> Path:
    """Return the directory holding generated reviewer previews."""
    return Path(output_root) / "review_labels" / "previews"


def preview_path(output_root: Path, source: Path, max_dimension: int) -> Path:
    """Return the deterministic cache path for one preview rendering."""
    token = hashlib.sha1(str(Path(source).resolve()).encode("utf-8")).hexdigest()[:16]
    return preview_cache_dir(output_root) / f"{token}_{max_dimension}.jpg"


def ensure_preview(
    output_root: Path,
    source: Path,
    max_dimension: int = EDIT_MAX_DIMENSION,
) -> Path | None:
    """Return a cached clean JPEG preview, generating it when missing."""
    source = Path(source)
    target = preview_path(output_root, source, max_dimension)
    if target.is_file() and target.stat().st_size > 0:
        return target
    if not source.is_file():
        return None
    decoded = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if decoded is None:
        LOGGER.warning("Could not decode source for preview: %s", source)
        return None
    height, width = decoded.shape[:2]
    scale = min(1.0, max_dimension / max(height, width, 1))
    if scale < 1.0:
        decoded = cv2.resize(
            decoded,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.jpg")
    encode_options = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    if not cv2.imwrite(str(temporary), decoded, encode_options):
        LOGGER.warning("Could not write preview for %s", source)
        return None
    temporary.replace(target)
    return target if target.is_file() and target.stat().st_size > 0 else None
