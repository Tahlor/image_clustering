"""Image decoding and local-feature extraction."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.models import ImageFeatures, ImageItem

_FEATURE_CACHE_VERSION = 2


def _read_gray(path: Path, max_dimension: int) -> tuple[np.ndarray, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    scale = min(1.0, max_dimension / float(max(image.shape)))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image, scale


def _cache_key(image: ImageItem, config: ClusterConfig) -> str:
    stat = image.path.stat()
    payload = (
        f"v{_FEATURE_CACHE_VERSION}|opencv={cv2.__version__}|numpy={np.__version__}|"
        f"{image.path.resolve()}|"
        f"{stat.st_size}|{stat.st_mtime_ns}|"
        f"{config.max_working_dimension}|{config.max_features}|"
        f"{config.sift_contrast_threshold}|"
        f"working_image={config.cache_working_images}|"
        f"compressed={config.feature_cache_compressed}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _legacy_cache_key(image: ImageItem, config: ClusterConfig) -> str:
    stat = image.path.stat()
    payload = (
        f"{image.path}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{config.max_working_dimension}|{config.max_features}|"
        f"{config.sift_contrast_threshold}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_save_cache(
    path: Path,
    *,
    gray: np.ndarray,
    scale: float,
    keypoints_xy: np.ndarray,
    descriptors: np.ndarray,
    config: ClusterConfig,
) -> None:
    values: dict[str, np.ndarray] = {
        "scale": np.asarray(scale, dtype=np.float64),
        "keypoints_xy": keypoints_xy,
        "descriptors": descriptors,
    }
    if config.cache_working_images:
        values["gray"] = gray

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            if config.feature_cache_compressed:
                np.savez_compressed(handle, **values)
            else:
                np.savez(handle, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cache(
    path: Path,
    *,
    image: ImageItem,
    config: ClusterConfig,
) -> ImageFeatures | None:
    try:
        with np.load(path, allow_pickle=False) as cached:
            keypoints_xy = cached["keypoints_xy"].copy()
            descriptors = cached["descriptors"].copy()
            scale = float(cached["scale"])
            gray = cached["gray"].copy() if "gray" in cached.files else None
    except (EOFError, KeyError, OSError, ValueError):
        path.unlink(missing_ok=True)
        return None

    if gray is None:
        gray, decoded_scale = _read_gray(
            path=image.path,
            max_dimension=config.max_working_dimension,
        )
        if not np.isclose(scale, decoded_scale):
            path.unlink(missing_ok=True)
            return None

    return ImageFeatures(
        image=image,
        gray=gray,
        scale=scale,
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
    )


def _upgrade_legacy_cache(
    path: Path,
    *,
    image: ImageItem,
    config: ClusterConfig,
    destination: Path,
) -> ImageFeatures | None:
    try:
        with np.load(path, allow_pickle=False) as cached:
            keypoints_xy = cached["keypoints_xy"].copy()
            descriptors = cached["descriptors"].copy()
    except (EOFError, KeyError, OSError, ValueError):
        return None

    gray, scale = _read_gray(
        path=image.path,
        max_dimension=config.max_working_dimension,
    )
    _atomic_save_cache(
        destination,
        gray=gray,
        scale=scale,
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
        config=config,
    )
    return ImageFeatures(
        image=image,
        gray=gray,
        scale=scale,
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
    )


def extract_features(
    image: ImageItem,
    config: ClusterConfig,
    cache_dir: Path | None = None,
) -> ImageFeatures:
    """Load or compute the exact working image and SIFT features for one source."""
    cache_path = None
    if config.cache_features and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_cache_key(image=image, config=config)}.npz"
        if cache_path.exists():
            cached = _load_cache(
                cache_path,
                image=image,
                config=config,
            )
            if cached is not None:
                return cached
        legacy_path = cache_dir / (
            f"{_legacy_cache_key(image=image, config=config)}.npz"
        )
        if legacy_path.exists():
            upgraded = _upgrade_legacy_cache(
                legacy_path,
                image=image,
                config=config,
                destination=cache_path,
            )
            if upgraded is not None:
                return upgraded

    gray, scale = _read_gray(
        path=image.path,
        max_dimension=config.max_working_dimension,
    )
    detector = cv2.SIFT_create(
        nfeatures=config.max_features,
        contrastThreshold=config.sift_contrast_threshold,
    )
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    keypoints_xy = np.float32([keypoint.pt for keypoint in keypoints])
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    if cache_path is not None:
        _atomic_save_cache(
            cache_path,
            gray=gray,
            scale=scale,
            keypoints_xy=keypoints_xy,
            descriptors=descriptors,
            config=config,
        )
    return ImageFeatures(
        image=image,
        gray=gray,
        scale=scale,
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
    )
